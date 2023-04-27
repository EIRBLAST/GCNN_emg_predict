import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import pickle, os, scipy.io, random
import copy

from nmf_emg.mapping import AlphaMatrix


# ********************************************
class NMF_Routine:
    @staticmethod
    def _nmf(X):
        model = NMF(
            n_components=2, init="random", random_state=None, solver="mu", max_iter=1000, beta_loss="kullback-leibler"
        )
        W = model.fit_transform(X)
        return W, model.components_  # latent

    @staticmethod
    def compute(X, n_rep=10, diff_signal=True):
        _, H = NMF_Routine._nmf(X)

        # repeat NMF routine n_rep times for better smooting
        for _ in range(n_rep):
            _, H = NMF_Routine._nmf(H)

        # check shape H high/low -> signal should be high -> low -> high
        if H[0, 0] - H[1, 0] < 0:
            H = H[[1, 0], :]  # swap

        # normalization of each signal
        Hn = (H.T / np.max(H, axis=1)).T  # 2 x timestep

        if diff_signal:
            S = Hn[1, :] - Hn[0, :]
        else:
            S = Hn

        # normalization 0-1
        Sn = (S - np.min(S)) / np.ptp(S)
        return Sn


# ********************************************
class Labeler:
    def __init__(self, data_path, alpha_type, diff_signal, n_rep_nmf):
        if "ar10" in alpha_type:
            self.alpha = AlphaMatrix.ar10
        elif "ub" in alpha_type:
            self.alpha = AlphaMatrix.ub
        elif "berrett" in alpha_type:
            self.alpha = AlphaMatrix.berrett
        else:
            NotImplementedError("AlphaMatrix type not available!")

        self.diff_signal = diff_signal
        self.n_rep_nmf = n_rep_nmf
        self.data_path = data_path

    def merge_1_signal(self, H_pw, H_pn, H_ul):
        # column 1 of T
        alpha1 = np.repeat(
            self.alpha[:, 0].reshape(-1, 1), repeats=H_pw.shape[-1], axis=1
        )  # repeat signal k times along columns to match h1 dims
        T1 = np.multiply(alpha1, np.tile(H_pw, (3, 1)))

        # column 2 of T
        alpha2_rep = np.repeat(self.alpha[:, 1].reshape(-1, 1), repeats=H_pn.shape[-1], axis=1)
        T2 = np.multiply(alpha2_rep, np.tile(H_pn, (3, 1)))

        # column 3 of T
        alpha3_rep = np.repeat(self.alpha[:, 2].reshape(-1, 1), repeats=H_ul.shape[-1], axis=1)
        T3 = np.multiply(alpha3_rep, np.tile(H_ul, (3, 1)))

        return np.hstack([T1, T2, T3])

    def merge_2_signals(self, H_pw, H_pn, H_ul):
        alpha_bar = np.zeros((self.alpha.shape[0] * 2, self.alpha.shape[1]))
        alpha_bar[0:2, :] = np.repeat(self.alpha[0, :].reshape(1, -1), repeats=2, axis=0)
        alpha_bar[2:4, :] = np.repeat(self.alpha[1, :].reshape(1, -1), repeats=2, axis=0)
        alpha_bar[-2:, :] = np.repeat(self.alpha[2, :].reshape(1, -1), repeats=2, axis=0)

        # column 1 of T
        alpha1 = np.repeat(
            alpha_bar[:, 0].reshape(-1, 1), repeats=H_pw.shape[-1], axis=1
        )  # repeat signal k times along columns to match h1 dims
        T1 = np.multiply(alpha1, np.tile(H_pw, (3, 1)))

        # column 2 of T
        alpha2_rep = np.repeat(alpha_bar[:, 1].reshape(-1, 1), repeats=H_pn.shape[-1], axis=1)
        T2 = np.multiply(alpha2_rep, np.tile(H_pn, (3, 1)))

        # column 3 of T
        alpha3_rep = np.repeat(alpha_bar[:, 2].reshape(-1, 1), repeats=H_ul.shape[-1], axis=1)
        T3 = np.multiply(alpha3_rep, np.tile(H_ul, (3, 1)))

        # T
        T = np.hstack([T1, T2, T3])
        print("T matrix: ", T.shape)
        return T

    def load_data(self, subj_id=1):
        # load mat files
        F_pw = np.array(
            scipy.io.loadmat(f"{self.data_path}/subj{subj_id}_pwr.mat")[f"subj{subj_id}_pwr"]
        ).squeeze()  # 6 x 1 -> 6 ripetizioni
        F_pn = np.array(scipy.io.loadmat(f"{self.data_path}/subj{subj_id}_pin.mat")[f"subj{subj_id}_pin"]).squeeze()
        F_ul = np.array(scipy.io.loadmat(f"{self.data_path}/subj{subj_id}_uln.mat")[f"subj{subj_id}_uln"]).squeeze()

        F_pw_ref = np.array(
            scipy.io.loadmat(f"{self.data_path}/subj{subj_id}_pwr_ref.mat")[f"subj{subj_id}_pwr_ref"]
        ).squeeze()  # 6 x 1 -> 6 ripetizioni
        F_pn_ref = np.array(
            scipy.io.loadmat(f"{self.data_path}/subj{subj_id}_pin_ref.mat")[f"subj{subj_id}_pin_ref"]
        ).squeeze()
        F_ul_ref = np.array(
            scipy.io.loadmat(f"{self.data_path}/subj{subj_id}_uln_ref.mat")[f"subj{subj_id}_uln_ref"]
        ).squeeze()
        return F_pw, F_pn, F_ul, F_pw_ref, F_pn_ref, F_ul_ref

    def run_subj(self, plot=True):
        F_pw, F_pn, F_ul, F_pw_ref, F_pn_ref, F_ul_ref = self.load_data()
        data_dict = {}
        for rep in range(len(F_pw)):
            print("-> rep =", rep)

            print("X | power: {}, pinch: {}, ulnar: {}".format(F_pw[rep].shape, F_pn[rep].shape, F_ul[rep].shape))
            print(
                "R | power: {}, pinch: {}, ulnar: {}".format(
                    F_pw_ref[rep].shape, F_pn_ref[rep].shape, F_ul_ref[rep].shape
                )
            )
            assert (
                F_pw[rep].shape[1] == F_pw_ref[rep].shape[1]
                and F_pn[rep].shape[1] == F_pn_ref[rep].shape[1]
                and F_ul[rep].shape[1] == F_ul_ref[rep].shape[1]
            )

            # *****************************
            # NMF LABELING
            H_pw = NMF_Routine.compute(F_pw[rep], n_rep=self.n_rep_nmf, diff_signal=self.diff_signal)  # power
            H_pn = NMF_Routine.compute(F_pn[rep], n_rep=self.n_rep_nmf, diff_signal=self.diff_signal)  # pinch
            H_ul = NMF_Routine.compute(F_ul[rep], n_rep=self.n_rep_nmf, diff_signal=self.diff_signal)  # ulnar

            # merge label signals
            if self.diff_signal:
                T = self.merge_1_signal(H_pw, H_pn, H_ul)
            else:
                T = self.merge_2_signals(H_pw, H_pn, H_ul)

            # emg input signals
            X = np.hstack([F_pw[rep], F_pn[rep], F_ul[rep]])

            # *****************************
            # REFERENCE SIGNAL

            # normalize REF to 0-1 and flip
            R_pw = 1 - (F_pw_ref[rep] - np.min(F_pw_ref[rep])) / np.ptp(F_pw_ref[rep])
            R_pn = 1 - (F_pn_ref[rep] - np.min(F_pn_ref[rep])) / np.ptp(F_pn_ref[rep])
            R_ul = 1 - (F_ul_ref[rep] - np.min(F_ul_ref[rep])) / np.ptp(F_ul_ref[rep])
            R = self.merge_1_signal(R_pw, R_pn, R_ul)

            # *****************************
            # NORMALIZE X, T and R
            Xn = (X - np.min(X)) / np.ptp(X)
            Tn = (T - np.min(T)) / np.ptp(T)
            Rn = (R - np.min(R)) / np.ptp(R)

            data_dict[rep] = {
                "X": copy.deepcopy(X),
                "T": copy.deepcopy(T),
                "R": copy.deepcopy(R),
                "Xn": copy.deepcopy(Xn),
                "Tn": copy.deepcopy(Tn),
                "Rn": copy.deepcopy(Rn),
            }

            if plot:
                self.plot(X, T, R, Xn, Tn, Rn)
        return data_dict

    def plot(self, X, T, R, Xn, Tn, Rn):
        fig, axs = plt.subplots(4, 2, figsize=(15, 8))

        axs[0, 0].set_title("Not normalized")
        axs[0, 1].set_title("Normalized")

        axs[0, 0].plot(X.T)

        axs[1, 0].plot(T[0, :].T)
        axs[1, 0].plot(R[0, :].T)

        axs[2, 0].plot(T[1, :].T)
        axs[2, 0].plot(R[1, :].T)

        axs[3, 0].plot(T[2, :].T)
        axs[3, 0].plot(R[2, :].T)

        axs[0, 1].plot(Xn.T)

        axs[1, 1].plot(Tn[0, :].T)
        axs[1, 1].plot(Rn[0, :].T)

        axs[2, 1].plot(Tn[1, :].T)
        axs[2, 1].plot(Rn[1, :].T)

        axs[3, 1].plot(Tn[2, :].T)
        axs[3, 1].plot(Rn[2, :].T)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    PLOT = False
    SEED = 1
    N_REP_NMF = 10
    ALPHA = "ar10"
    DATA_PATH = "data"
    OUTPUT_PATH = "datasets"

    np.random.seed(SEED)
    random.seed(SEED)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    labeler = Labeler(data_path=DATA_PATH, alpha_type=ALPHA, diff_signal=True, n_rep_nmf=N_REP_NMF)
    data = labeler.run_subj(plot=PLOT)

    with open(f"{OUTPUT_PATH}/subj_{ALPHA}.pickle", "wb") as handle:
        pickle.dump(data, handle)
