import numpy as np
from typing import List, Tuple
from src.utils import LabelEncoder

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

        pi_total = np.sum(self.pi)
        if pi_total == 0:
            # No data; fall back to uniform to avoid NaNs.
            self.pi[:] = 1.0 / self.num_states
        else:
            self.pi /= pi_total

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
        eps = 1e-15
        T = len(observation_sequence)
        L = np.zeros((self.num_states, T))

        #l_i0 = log(pi_i) + log(b_i0)
        for i in range(self.num_states):
            L[i][0] = np.log(self.pi[i] + eps) + np.log(self.B[i][observation_sequence[0]] + eps)
        
        #l_j,t+1 = max_i[l_it + log(a_ij)] + log(b_j,t+1)
        for i in range(1, T):
            observation = observation_sequence[i]
            for j in range(self.num_states):
                totalMax = -np.inf
                for k in range(self.num_states):
                    totalMax = max(L[k][i - 1] + np.log(self.A[k][j] + eps), totalMax)
                L[j][i] = totalMax + np.log(self.B[j][observation] + eps)
        
        stateSequence = []
        stateSequence.append(np.argmax(L[:, T - 1]))

        #s*_t = argmax[l*_it + log(a_i,s*_t+1)]
        for t in range(T - 2, -1, -1):
            totalMax = -np.inf
            maxState = 0
            for i in range(self.num_states):
                current = L[i][t] + np.log(self.A[i][stateSequence[-1]] + eps)
                if current > totalMax:
                    totalMax = current
                    maxState = i
            stateSequence.append(maxState)

        return stateSequence[::-1]


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
        """
        Initialize A, B, and pi with random probabilities (normalized).
        """
        np.random.seed(random_seed)

        # Initialize self.A (random, normalize rows to sum to 1)
        for i in range(self.num_states):
            for j in range(self.num_states):
                for k in range(self.num_states):
                    self.A[i][j][k] = np.random.rand()
                self.A[i][j] = self.A[i][j] / np.sum(self.A[i][j])

        # Initialize self.B (random, normalize rows to sum to 1)
        for i in range(self.num_states):
            for j in range(self.num_observations):
                self.B[i][j] = np.random.rand()
            self.B[i] = self.B[i] / np.sum(self.B[i])

        # Initialize self.pi (random, normalize to sum to 1)
        for i in range(self.num_states):
            for j in range(self.num_states):
                self.pi[i][j] = np.random.rand()
        self.pi = self.pi / np.sum(self.pi)

    def maximum_likelihood_initialize_parameters(self, train_sequences: List[List[int]], purpose_encoder: "LabelEncoder", mode_encoder: "LabelEncoder"):
        """
        Initialize A, B, pi using Maximum Likelihood on the training data.
        """
        self.A.fill(0.0)
        self.B.fill(0.0)
        self.pi.fill(0.0)

        for seq in train_sequences:
            len_seq = len(seq)
            if len_seq == 0:
                continue

            prev = seq[0]
            self.B[purpose_encoder.transform([prev[1]])[0]][mode_encoder.transform([prev[0]])[0]] += 1

            if len_seq == 1:
                continue

            prev2 = seq[1]
            self.B[purpose_encoder.transform([prev2[1]])[0]][mode_encoder.transform([prev2[0]])[0]] += 1
            self.pi[purpose_encoder.transform([prev[1]])[0]][purpose_encoder.transform([prev2[1]])[0]] += 1

            if len_seq == 2:
                continue

            for i in range(2, len_seq):
                prev3 = seq[i]
                self.A[purpose_encoder.transform([prev[1]])[0]][purpose_encoder.transform([prev2[1]])[0]][purpose_encoder.transform([prev3[1]])[0]] += 1
                self.B[purpose_encoder.transform([prev3[1]])[0]][mode_encoder.transform([prev3[0]])[0]] += 1
                prev = prev2
                prev2 = prev3

        self.pi = self.pi / np.sum(self.pi)
        A_row_sums = self.A.sum(axis=2, keepdims=True)
        np.divide(self.A, A_row_sums, out=self.A, where=A_row_sums != 0)
        B_row_sums = self.B.sum(axis=1, keepdims=True)
        np.divide(self.B, B_row_sums, out=self.B, where=B_row_sums != 0)

    def predict_viterbi(self, observation_sequence: List[int]) -> List[int]:
        """
        Decode the most likely sequence of hidden states using the Viterbi algorithm.
        """
        eps = 1e-15
        T = len(observation_sequence)
        L = np.zeros((self.num_states, self.num_states, T))

        if T == 1:
            bestState = 0
            bestProbability = -np.inf
            observation = observation_sequence[0]
            for i in range(self.num_states):
                p = np.log(self.B[i][observation] + eps)
                if p > bestProbability:
                    bestProbability = p
                    bestState = i
            return [bestState]

        if T == 2:
            totalMax = -np.inf
            besti = 0
            bestj = 0
            o0 = observation_sequence[0]
            o1 = observation_sequence[1]
            for i in range(self.num_states):
                for j in range(self.num_states):
                    v = np.log(self.pi[i][j] + eps) + np.log(self.B[i][o0] + eps) + np.log(self.B[j][o1] + eps)
                    if v > totalMax:
                        totalMax = v
                        besti = i
                        bestj = j
            return [besti, bestj]

        for i in range(self.num_states):
            for j in range(self.num_states):
                L[i][j][1] = (
                    np.log(self.pi[i][j] + eps)
                    + np.log(self.B[i][observation_sequence[0]] + eps)
                    + np.log(self.B[j][observation_sequence[1]] + eps)
                )

        for t in range(2, T):
            observation = observation_sequence[t]
            for i in range(self.num_states):
                for j in range(self.num_states):
                    totalMax = -np.inf
                    for k in range(self.num_states):
                        totalMax = max(L[k][i][t - 1] + np.log(self.A[k][i][j] + eps), totalMax)
                    L[i][j][t] = totalMax + np.log(self.B[j][observation] + eps)

        stateSequence = []

        totalMax = -np.inf
        lasti = 0
        lastj = 0
        for i in range(self.num_states):
            for j in range(self.num_states):
                if L[i][j][T - 1] > totalMax:
                    totalMax = L[i][j][T - 1]
                    lasti = i
                    lastj = j

        stateSequence.append(lastj)
        stateSequence.append(lasti)

        for t in range(T - 1, 1, -1):
            totalMax = -np.inf
            maxState = 0
            prev_i = stateSequence[-1]
            prev_j = stateSequence[-2]
            for i in range(self.num_states):
                current = L[i][prev_i][t - 1] + np.log(self.A[i][prev_i][prev_j] + eps)
                if current > totalMax:
                    totalMax = current
                    maxState = i
            stateSequence.append(maxState)

        return stateSequence[::-1]


class EdgeEmittingHiddenMarkovModel:
    """
    Edge-emitting HMM for trip purpose modeling.

    - Hidden states: trip purposes (encoded as integers 0..K-1)
    - Observations: modes (encoded as integers 0..V-1)
    - START state: index K (virtual previous state for t=0)

    Parameters:

        num_states (int):      Number of hidden states (purposes), K.
        num_observations (int):Number of possible observations (modes), V.

    Notes:
        - training data uses ML , since purpose are known
    """

    def __init__(self, num_states: int, num_observations: int, smoothing: float = 1.0):
        self.num_states = num_states     
        self.num_observations = num_observations  
        self.START_STATE = num_states   
        self.smoothing = smoothing   

        # Transition matrix A: (K+1, K)
        self.A = np.zeros((self.num_states + 1, self.num_states), dtype=float)

        # Edge emission matrix B: (K+1, K, V)
        self.B = np.zeros(
            (self.num_states + 1, self.num_states, self.num_observations),
            dtype=float,
        )

    def randomly_initialize_parameters(self, random_seed: int = 42):
        """
        Randomly initialize A and B with valid probability distributions.
        """
        np.random.seed(random_seed)

        # --- Initialize A: row-wise normalization ---
        for i in range(self.num_states + 1):  # include START row
            for j in range(self.num_states):
                self.A[i][j] = np.random.rand()
            self.A[i] = self.A[i]/np.sum(self.A[i])
            

        # --- Initialize B: normalize over 'obs' dimension ---
        for i in range(self.num_states + 1):      
            for j in range(self.num_states):     
                for v in range(self.num_observations):
                    self.B[i][j][v] = np.random.rand()
                self.B[i][j] = self.B[i][j]/np.sum(self.B[i][j])
                

    def maximum_likelihood_initialize_parameters(
        self,
        train_sequences: List[List[Tuple[str, str, str]]],
        purpose_encoder: "LabelEncoder",
        mode_encoder: "LabelEncoder",
    ):
        """
        Estimate A and B via supervised Maximum Likelihood.

        Args:
            train_sequences:
                List of daily sequences.
                Each sequence is a list of tuples:
                    (mode_str, purpose_str, timestamp)

            purpose_encoder:
                LabelEncoder already fitted on all purposes
                transform(purpose_str) -> int in [0..K-1]

            mode_encoder:
                LabelEncoder already fitted on all modes
                transform(mode_str) -> int in [0..V-1]

        """
        # 1) reset counts
        self.A.fill(0.0)
        self.B.fill(0.0)

        START = self.START_STATE
        # 2) accumulate counts from data
        for seq in train_sequences:
            if len(seq) == 0:
                continue

            # every seq is a (mode_str, purpose_str, timestamp) tuple
            modes = [row[0] for row in seq]
            purposes = [row[1] for row in seq]

            # transform
            state_indices = purpose_encoder.transform(purposes)  # shape (T,)
            obs_indices = mode_encoder.transform(modes)          # shape (T,)

            T = len(state_indices)
            if T == 0:
                continue

            z0 = state_indices[0] #Purpose
            x0 = obs_indices[0] # mode

            # first traj
            self.A[START,z0] +=1.0
            self.B[START,z0,x0] +=1.0
            
            
            for t in range(1,T):
                prev_state = state_indices[t-1]
                curr_state = state_indices[t]
                x_t = obs_indices[t]
                self.A[prev_state,curr_state] +=1.0
                self.B[prev_state,curr_state,x_t] +=1.0


        # 3) Laplace smoothing + normalization

        alpha = self.smoothing                
        A_smooth = self.A + alpha

        row_sums = A_smooth.sum(axis = 1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.A = A_smooth/row_sums

        B_smooth = self.B + alpha 
        obs_sums = B_smooth.sum(axis = 2 , keepdims=True)
        obs_sums [obs_sums ==0] = 1.0
        self.B = B_smooth/obs_sums


    def predict_viterbi(self, observation_sequence: List[int]) -> List[int]:
        T = len(observation_sequence)
        if T == 0:
            return []

        K = self.num_states
        START = self.START_STATE

        eps = 1e-15
        logA = np.log(self.A + eps)        # (K+1, K)
        logB = np.log(self.B + eps)        # (K+1, K, V)

        dp = np.full((T, K), -np.inf)
        backptr = np.full((T, K), -1, dtype=int)

        # Initialization at t = 0
        x0 = observation_sequence[0]
        for j in range(K):
            dp[0, j] = logA[START, j] + logB[START, j, x0]
            backptr[0, j] = START

        # Recursion for t = 1..T-1
        for t in range(1, T):
            x_t = observation_sequence[t]
            for j in range(K):
                # scores from all previous states i -> j
                scores = dp[t - 1, :] + logA[:K, j] + logB[:K, j, x_t]
                i_star = int(np.argmax(scores))
                dp[t, j] = scores[i_star]
                backptr[t, j] = i_star

        # Backtrace
        last_state = int(np.argmax(dp[T - 1, :]))
        state_sequence = [last_state]

        for t in range(T - 2, -1, -1):
            prev_state = backptr[t + 1, state_sequence[-1]]
            state_sequence.append(prev_state)

        state_sequence.reverse()
        return state_sequence

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