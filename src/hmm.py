import numpy as np
from typing import List, Tuple
from src.utils import LabelEncoder

class ThreeGramHiddenMarkovModel:
    """
    Custom implementation of a 3-gram Hidden Markov Model (IN PROGRESS)
    """
    def __init__(self, num_states: int, num_observations: int):
        self.num_states = num_states
        self.num_observations = num_observations
        self.A = np.zeros((num_states, num_states, num_states), dtype=float)
        self.B = np.zeros((num_states, num_observations), dtype=float)
        self.pi = np.zeros((num_states, num_states), dtype=float)

    def randomly_initialize_parameters(self, random_seed: int = 42):
        pass

    def maximum_likelihood_initialize_parameters(self, train_sequences: List[List[int]], purpose_encoder: "LabelEncoder", mode_encoder: "LabelEncoder"):
        for seq in train_sequences:
            # Here the length of the sequence matters
            len_seq = len(seq)
            prev = seq[0]
            self.B[purpose_encoder.transform([prev[1]])[0]][mode_encoder.transform([prev[0]])[0]] += 1

            if len_seq == 1:
                # we can only update the emission probabilities
                continue

            prev2 = seq[1]
            self.B[purpose_encoder.transform([prev2[1]])[0]][mode_encoder.transform([prev2[0]])[0]] += 1
            self.pi[purpose_encoder.transform([prev[1]])[0]][purpose_encoder.transform([prev2[1]])[0]] += 1

            if len_seq == 2:
                # we can update the emission and initial probabilities
                continue
            
            # we can update the emission, initial, and 3-gram transition probabilities
            for i in range(2, len_seq):
                prev3 = seq[i]
                self.A[purpose_encoder.transform([prev[1]])[0]][purpose_encoder.transform([prev2[1]])[0]][purpose_encoder.transform([prev3[1]])[0]] += 1
                self.B[purpose_encoder.transform([prev3[1]])[0]][mode_encoder.transform([prev3[0]])[0]] += 1
                prev = prev2
                prev2 = prev3

            self.pi = self.pi / np.sum(self.pi)
            # normalize A along last axis (sum_k A[i,j,k] = 1 when row has data)
            A_row_sums = self.A.sum(axis=2, keepdims=True)   # shape: (num_states, num_states, 1)
            np.divide(self.A, A_row_sums, out=self.A, where=A_row_sums != 0)

            # normalize B along last axis (sum_o B[k,o] = 1 when row has data)
            B_row_sums = self.B.sum(axis=1, keepdims=True)   # shape: (num_states, 1)
            np.divide(self.B, B_row_sums, out=self.B, where=B_row_sums != 0)




class HiddenMarkovModel:
    """
    Custom implementation of a Hidden Markov Model.
    
    Attributes:
        num_states (int): Number of hidden states (purposes).
        num_observations (int): Number of possible observations (modes).
        A (np.ndarray): Transition probability matrix (num_states x num_states).
                        A[i, j] = P(state_t+1 = j | state_t = i)
        B (np.ndarray): Emission probability matrix (num_states x num_observations).
                        B[i, k] = P(observation_t = k | state_t = i)
        pi (np.ndarray): Initial state probability distribution (num_states,).
                        pi[i] = P(state_0 = i)
    """
    
    def __init__(self, num_states: int, num_observations: int):
        self.num_states = num_states
        self.num_observations = num_observations
        
        self.A = np.zeros((num_states, num_states), dtype=float)
        self.B = np.zeros((num_states, num_observations), dtype=float)
        self.pi = np.zeros(num_states, dtype=float)
        
    def randomly_initialize_parameters(self, random_seed: int = 42):
        """
        Initialize A, B, and pi with random probabilities (normalized).
        """
        np.random.seed(random_seed)
        
        # Initialize self.A (random, normalize rows to sum to 1)
        for i in range(self.num_states):
            for j in range(self.num_states):
                self.A[i][j] = np.random.rand()
            self.A[i] = self.A[i] / np.sum(self.A[i])

        # Initialize self.B (random, normalize rows to sum to 1)
        for i in range(self.num_states):
            for j in range(self.num_observations):
                self.B[i][j] = np.random.rand()
            self.B[i] = self.B[i] / np.sum(self.B[i])
        
        # Initialize self.pi (random, normalize to sum to 1)
        for i in range(self.num_states):
            self.pi[i] = np.random.rand()
        self.pi = self.pi / np.sum(self.pi)

    """
    Following a discussion with our TA mentor after project milestone 2, it became apparent that because the hidden variables in this modeling problem (the trip purposes) are fully observed in the training data, we can use Maximum Likelihood to train the HMM."""

    def maximum_likelihood_initialize_parameters(self, train_sequences: List[List[int]], purpose_encoder: "LabelEncoder", mode_encoder: "LabelEncoder"):
        """
        Initialize A, B, pi using Maximum Likelihood on the training data.
        """
        # Reset parameters to zero before counting
        self.A.fill(0.0)
        self.B.fill(0.0)
        self.pi.fill(0.0)
        # Intuition tells me that we can just populate the counts and normalize last.
        for seq in train_sequences:
            if len(seq) == 0:
                continue
            prev = seq[0]
            # every element of seq has structure (mode, purpose, timestamp, timezone)
            # we can increment the count of pi with head[1]
            self.pi[purpose_encoder.transform([prev[1]])[0]] += 1
            self.B[purpose_encoder.transform([prev[1]])[0]][mode_encoder.transform([prev[0]])[0]] += 1

            for i in range(1, len(seq)):
                curr = seq[i]
                self.A[purpose_encoder.transform([prev[1]])[0]][purpose_encoder.transform([curr[1]])[0]] += 1
                self.B[purpose_encoder.transform([curr[1]])[0]][mode_encoder.transform([curr[0]])[0]] += 1
                prev = curr

        self.pi = self.pi / np.sum(self.pi)
        row_sums = self.A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.A /= row_sums
        row_sums = self.B.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.B /= row_sums

    def _forward(self, observation_sequence: List[int]) -> np.ndarray:
        """
        Compute the alpha values (forward probabilities).
        
        alpha[i, t] = P(o_1, ..., o_t, q_t = i | lambda)
        
        Args:
            observation_sequence: List of integer indices representing observations.
            
        Returns:
            alpha: Matrix of shape (num_states, T).
        """
        T = len(observation_sequence)
        alpha = np.zeros((self.num_states, T))
        
        #alpha_i0 = pi_i * b_i0
        for i in range(self.num_states):
            alpha[i][0] = self.pi[i] * self.B[i][observation_sequence[0]]

        #alpha_j,t+1 = sum (alpha_it * a_ij * b_j,t+1)
        for i in range(1, T):
            observation = observation_sequence[i]
            for j in range(self.num_states):
                totalSum = 0
                for k in range(self.num_states):
                    totalSum += alpha[k][i - 1] * self.A[k][j] * self.B[j][observation]
                alpha[j][i] = totalSum
        return alpha

    def _backward(self, observation_sequence: List[int]) -> np.ndarray:
        """
        Compute the beta values (backward probabilities).
        
        beta[t, i] = P(o_t+1, ..., o_T | q_t = i, lambda)
        
        Args:
            observation_sequence: List of integer indices representing observations.
            
        Returns:
            beta: Matrix of shape (T, num_states).
        """
        T = len(observation_sequence)
        beta = np.zeros((T, self.num_states))
        
        # TODO: Initialization (t=T-1)
        # beta[T-1, i] = 1
        
        # TODO: Induction (t=T-2 to 0)
        # beta[t, i] = sum_j(A[i, j] * B[j, O_t+1] * beta[t+1, j])
        
        return beta

    def fit_baum_welch(self, sequences: List[List[int]], n_iter: int = 10):
        """
        Train the HMM parameters using the Baum-Welch algorithm (EM).
        
        Args:
            sequences: List of observation sequences (each is a list of ints).
            n_iter: Number of EM iterations.
        """
        for iteration in range(n_iter):
            # Initialize accumulators for A_numer, A_denom, B_numer, B_denom, pi_accum
            
            for seq in sequences:
                # 1. E-Step: Compute forward (alpha) and backward (beta) probabilities
                # alpha = self._forward(seq)
                # beta = self._backward(seq)
                
                # Compute gamma (posterior probability of being in state i at time t)
                # gamma[t, i] = (alpha[t, i] * beta[t, i]) / P(O | lambda)
                
                # Compute xi (joint posterior of transitioning from i to j at time t)
                # xi[t, i, j] = (alpha[t, i] * A[i, j] * B[j, O_t+1] * beta[t+1, j]) / P(O | lambda)
                
                # Accumulate sufficient statistics for parameter updates
                pass
                
            # 2. M-Step: Update A, B, pi using accumulated values
            # self.A = ...
            # self.B = ...
            # self.pi = ...
            
            pass

    def predict_viterbi(self, observation_sequence: List[int]) -> List[int]:
        """
        Decode the most likely sequence of hidden states using the Viterbi algorithm.
        
        Args:
            observation_sequence: List of integer indices representing observations.
            
        Returns:
            List[int]: Most likely sequence of state indices.
        """
        T = len(observation_sequence)
        L = np.zeros((self.num_states, T))

        #l_i0 = log(pi_i) + log(b_i0)
        for i in range(self.num_states):
            L[i][0] = np.log(self.pi[i]) + np.log(self.B[i][observation_sequence[0]])
        
        #l_j,t+1 = max_i[l_it + log(a_ij)] + log(b_j,t+1)
        for i in range(1, T):
            observation = observation_sequence[i]
            for j in range(self.num_states):
                totalMax = -np.inf
                for k in range(self.num_states):
                    totalMax = max(L[k][i - 1] + np.log(self.A[k][j]), totalMax)
                L[j][i] = totalMax + np.log(self.B[j][observation])
        
        stateSequence = []
        stateSequence.append(np.argmax(L[:, T - 1]))

        #s*_t = argmax[l*_it + log(a_i,s*_t+1)]
        for t in range(T - 2, -1, -1):
            totalMax = -np.inf
            maxState = 0
            for i in range(self.num_states):
                current = L[i][t] + np.log(self.A[i][stateSequence[-1]])
                if current > totalMax:
                    totalMax = current
                    maxState = i
            stateSequence.append(maxState)

        return stateSequence[::-1]

if __name__ == "__main__":
    num_states = 2
    num_observations = 3
    hmm = HiddenMarkovModel(num_states, num_observations)

    hmm.A = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])

    hmm.B = np.array([
        [0.5, 0.4, 0.1],
        [0.1, 0.3, 0.6]
    ])

    hmm.pi = np.array([0.6, 0.4])

    #Observation sequence
    seq = [0, 1, 2]

    print("TEST: Forward Algorithm")
    alpha = hmm._forward(seq)
    print(alpha)
    print("P =", np.sum(alpha[:, -1]))

    print("TEST: Viterbi Decoding")
    viterbi_path = hmm.predict_viterbi(seq)
    print("Most likely state path:", viterbi_path)