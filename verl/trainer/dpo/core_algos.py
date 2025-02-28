import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    """
    Compute DPO loss as described in the paper "Direct Preference Optimization: 
    Your Language Model is Secretly a Reward Model"
    
    Args:
        policy_chosen_logps: Log probs from policy model for chosen responses
        policy_rejected_logps: Log probs from policy model for rejected responses
        reference_chosen_logps: Log probs from reference model for chosen responses
        reference_rejected_logps: Log probs from reference model for rejected responses
        beta: Temperature parameter for the DPO loss
        
    Returns:
        DPO loss and advantages
    """
    # Compute the log ratios between policy and reference model
    print(f'Shape of policy_chosen_logps: {policy_chosen_logps.shape}')
    print(f'Shape of reference_chosen_logps: {reference_chosen_logps.shape}')
    print(f'Shape of policy_rejected_logps: {policy_rejected_logps.shape}')
    print(f'Shape of reference_rejected_logps: {reference_rejected_logps.shape}')
    chosen_ratio = policy_chosen_logps - reference_chosen_logps
    rejected_ratio = policy_rejected_logps - reference_rejected_logps
    
    # Compute the implied reward
    logits = beta * (chosen_ratio - rejected_ratio)
    
    # Compute the DPO loss (negative log sigmoid of the logits)
    losses = -F.logsigmoid(logits)
    
    return losses.mean(), logits