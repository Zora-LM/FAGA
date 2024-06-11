import torch.nn as nn
import torch

class TransE(nn.Module):
    """`Translating Embeddings for Modeling Multi-relational Data`_ (TransE), which represents the relationships as translations in the embedding space.

    Attributes:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
        embedding_range: Uniform distribution range.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim].
        rel_emb: Relation embedding, shape:[num_rel, emb_dim].

    .. _Translating Embeddings for Modeling Multi-relational Data: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela
    """

    def __init__(self, dim):
        super(TransE, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, head_emb, relation_emb, tail_emb):
        """Calculating the score of triples.

        The formula for calculating the score is :math:`\gamma - ||h + r - t||_F`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.

        Returns:
            score: The score of triples.
        """
        score = (head_emb + relation_emb) - tail_emb
        output = self.sigmoid(score.sum(-1))

        return output


