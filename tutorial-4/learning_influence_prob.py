import numpy as np
from copy import copy

def simulate_episode(init_prob_matrix, max_steps):
    prob_matrix=init_prob_matrix.copy()
    n_nodes=prob_matrix.shape[0]
    initial_active_nodes=np.random.binomial(1, 0.2, size=(n_nodes))
    history=np.array([initial_active_nodes])
    active_nodes=initial_active_nodes
    newly_active_nodes=active_nodes
    t=0

    while (t<max_steps and np.sum(newly_active_nodes)>0):
        p = (prob_matrix.T*active_nodes).T
        activated_edges=p>np.random.rand(p.shape[0], p.shape[1])
        #remove values of activated edges from prob_matrix
        prob_matrix=prob_matrix*((p!=0)==activated_edges)

        #update active nodes
        newly_active_nodes=(np.sum(activated_edges, axis=0)>0)*(1-active_nodes)
        active_nodes=np.array(active_nodes+newly_active_nodes)
        history=np.concatenate((history, [newly_active_nodes]), axis=0)
        t+=1
    return history

def estimate_probabilities(dataset, node_index, n_nodes):
    estimated_probabilities=np.ones(n_nodes)*1.0/(n_nodes-1)
    print('Initial estimated probabilities: ', estimated_probabilities)
    credits_=np.zeros(n_nodes)
    occur_v_active=np.zeros(n_nodes)
    n_episodes=len(dataset)

    for episode in dataset:
        idx_w_active=np.argwhere(episode[:,node_index]==1).reshape(-1)
    
        if len(idx_w_active)>0 and idx_w_active>0:
            active_nodes_in_previous_step=episode[idx_w_active-1, :].reshape(-1)
            credits_ += active_nodes_in_previous_step / np.sum(active_nodes_in_previous_step)

        for v in range(n_nodes):
            if v!=node_index:
                idx_v_active=np.argwhere(episode[:, v]==1).reshape(-1)
                if len(idx_v_active) > 0 and (len(idx_w_active) == 0 or idx_v_active < idx_w_active):
                    occur_v_active[v]+=1
    estimated_probabilities=credits_/occur_v_active
    estimated_probabilities=np.nan_to_num(estimated_probabilities)

    return estimated_probabilities


n_nodes=5
n_episodes=30000
prob_matrix=np.random.uniform(0,0.1, size=(n_nodes, n_nodes))
node_index=3
dataset=[]

for e in range(n_episodes):
    dataset.append(simulate_episode(prob_matrix, 10))

estimated_probabilities=estimate_probabilities(dataset, node_index, n_nodes)

print('True probabilities: ', prob_matrix[node_index, :])
print('Estimated probabilities: ', estimated_probabilities)
