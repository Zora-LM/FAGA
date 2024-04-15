import torch.nn as nn
import torch


class RotatE(nn.Module):
    """`RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space`_ (RotatE), which defines each relation as a rotation from the source entity to the target entity in the complex vector space.

    Attributes:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
        embedding_range: Uniform distribution range.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim * 2].
        rel_emb: Relation_embedding, shape:[num_rel, emb_dim].

    .. _RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space: https://openreview.net/forum?id=HkgEQnRqYQ
    """
    def __init__(self, args):
        super(RotatE, self).__init__()
        self.args = args
        self.epsilon = 2.0
        self.embedding_range = nn.Parameter(torch.Tensor([(self.args.margin + self.epsilon) / self.args.emb_dim]),
                                            requires_grad=False)
        self.fc = nn.Linear(self.args.hid_dim_ent//2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, head_emb, relation_emb, tail_emb):
        """Calculating the score of triples.

        The formula for calculating the score is :math:`\gamma - \|h \circ r - t\|`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.

        Returns:
            score: The score of triples.
        """
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation_emb / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        output = self.sigmoid(self.fc(score))
        return output


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
        self.fc = nn.Identity()
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
        output = self.sigmoid(self.fc(score).sum(-1))

        return output


class DistMult(nn.Module):
    """`Embedding Entities and Relations for Learning and Inference in Knowledge Bases`_ (DistMult)

    Attributes:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
        embedding_range: Uniform distribution range.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim].
        rel_emb: Relation_embedding, shape:[num_rel, emb_dim].

    .. _Embedding Entities and Relations for Learning and Inference in Knowledge Bases: https://arxiv.org/abs/1412.6575
    """

    def __init__(self, args):
        super(DistMult, self).__init__()
        self.args = args
        self.fc = nn.Linear(self.args.hid_dim_ent, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, head_emb, relation_emb, tail_emb):
        """Calculating the score of triples.

        The formula for calculating the score is :math:`h^{\top} \operatorname{diag}(r) t`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.

        Returns:
            score: The score of triples.
        """

        score = (head_emb * relation_emb) * tail_emb
        output = self.sigmoid(self.fc(score))

        return output

class DistMult2(nn.Module):
    """`Embedding Entities and Relations for Learning and Inference in Knowledge Bases`_ (DistMult)

    Attributes:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
        embedding_range: Uniform distribution range.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim].
        rel_emb: Relation_embedding, shape:[num_rel, emb_dim].

    .. _Embedding Entities and Relations for Learning and Inference in Knowledge Bases: https://arxiv.org/abs/1412.6575
    """

    def __init__(self, args):
        super(DistMult2, self).__init__()
        self.args = args
        self.W = nn.Linear(self.args.hid_dim_ent, self.args.hid_dim_ent, bias=False)
        self.fc = nn.Linear(self.args.hid_dim_ent, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, head_emb, relation_emb, tail_emb):
        """Calculating the score of triples.

        The formula for calculating the score is :math:`h^{\top} \operatorname{diag}(r) t`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.

        Returns:
            score: The score of triples.
        """

        score = (head_emb * self.W(relation_emb)) * tail_emb
        output = self.sigmoid(self.fc(score))

        return output

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.fc = nn.Linear(self.args.hid_dim_ent*2 + self.args.hid_dim_rel, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, head_emb, relation_emb, tail_emb):

        h = torch.cat([head_emb, relation_emb, tail_emb], dim=-1)
        output = self.sigmoid(self.fc(h))

        return output

