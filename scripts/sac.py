
import torch
import torch.nn.functional as F
from torch.optim import Adam

from two_layers import QNetwork, Policy
from utils import Networks, Models, ReplayBuffer


class SAC(Networks, Models):
    def __init__(self, state_dim, action_dim, 
                 buffer_size,
                 gamma=0.99, 
                 tau=1e-2, 
                 alpha=0.2, 
                 hidden_dim=256,
                 lr=0.0003):
        super().__init__("SAC")
        # Params
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.lr=lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        #Log params
        self.save_critic_loss = None
        self.save_policy_loss = None
        self.save_alpha_loss = None        

        
        # Networks
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(device=self.device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(device=self.device)

        self.policy = Policy(state_dim, action_dim, hidden_dim).to(device=self.device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device=self.device)).item()
        
        self.hard_update(self.critic_target, self.critic)

        self.memory = ReplayBuffer(buffer_size) 

        #Optimizers
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr) 
        self.log_alpha_optim = Adam([self.log_alpha], lr=self.lr) 
        
    def get_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
            action = torch.tanh(action)
        return action.detach().cpu().numpy()[0]
    
    
    
    
    def update(self, batch_size):
        # Sample a batch from memory
        state, actions, reward, next_state, done = self.memory.sample(batch_size=batch_size)

        #Tensores
        state = torch.FloatTensor(state).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        #Critic loss
        with torch.no_grad():
            next_actions, next_log_pi, _, _ = self.policy.sample(next_state)
            next_q1, next_q2 = self.critic_target(next_state, next_actions)
            next_Q = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
            Qprime = reward + ((1 - done) * self.gamma * next_Q)

        qf1, qf2 = self.critic(state, actions)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, Qprime) # 
        qf2_loss = F.mse_loss(qf2, Qprime) # 
        critic_loss = qf1_loss + qf2_loss

        self.save_critic_loss = critic_loss.item()

        #Policy loss
        pi, log_pi, mean, log_std = self.policy.sample(state)

        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 
        self.save_policy_loss = policy_loss.item()

        #Alpha loss
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.save_alpha_loss = alpha_loss.item()

        #Update networks
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.log_alpha_optim.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        self.soft_update(self.critic_target, self.critic, self.tau)

    def __str__(self) -> str:
        return "SAC"

    
    