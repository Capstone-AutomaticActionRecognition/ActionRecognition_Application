import torch


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dist(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch]
