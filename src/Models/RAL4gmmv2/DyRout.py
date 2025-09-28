import torch
import torch.nn as nn

class DynamicRouting(nn.Module):
    def __init__(self, input_dim, hidden_dim):

        super(DynamicRouting, self).__init__()

        # Multilayer Perceptron (MLP) is used for information fusion.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, text_tokens, sep_token):
        """
        :param text_tokens:  [batch_size, num_tokens, input_dim]
        :param sep_token: [sep] [batch_size, input_dim]
        :return: the enhanced text representation [batch_size, num_tokens, input_dim]
        """
        batch_size, num_tokens, input_dim = text_tokens.size()    
        sep_token = sep_token.unsqueeze(1).expand(batch_size, num_tokens, input_dim)  # Duplicate the [sep] token so that it can be concatenated with each text token.
        # Concatenate the text tokens and the duplicated [sep] tokens
        concatenated_tokens = torch.cat((text_tokens, sep_token), dim=-1)
        # Perform information fusion via MLP
        enhanced_tokens = self.mlp(concatenated_tokens)

        return enhanced_tokens

