"""Flash Card Features Pro Max

Revision ID: a98f60393d04
Revises: 0a88133ba5e5
Create Date: 2025-06-01 07:38:15.912445

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a98f60393d04'
down_revision: Union[str, None] = '0a88133ba5e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('deck_card', sa.Column('question', sa.Text(), nullable=True))
    op.add_column('deck_card', sa.Column('answer', sa.Text(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('deck_card', 'answer')
    op.drop_column('deck_card', 'question')
    # ### end Alembic commands ###
