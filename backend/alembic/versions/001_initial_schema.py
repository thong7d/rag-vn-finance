"""initial_schema

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-09-25 20:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. chat_sessions
    op.create_table(
        'chat_sessions',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=100), nullable=False),
        sa.Column('first_question', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chat_sessions_user_id'), 'chat_sessions', ['user_id'], unique=False)

    # 2. chat_messages
    op.create_table(
        'chat_messages',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('session_id', sa.String(length=36), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('answer', sa.Text(), nullable=False),
        sa.Column('model_used', sa.String(length=50), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('decompose_enabled', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('sub_queries', sa.JSON(), nullable=True),
        sa.Column('source_ids', sa.JSON(), nullable=True),
        sa.Column('source_scores', sa.JSON(), nullable=True),
        sa.Column('source_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chat_messages_session_id'), 'chat_messages', ['session_id'], unique=False)
    op.create_index(op.f('ix_chat_messages_created_at'), 'chat_messages', ['created_at'], unique=False)

    # 3. user_feedbacks
    op.create_table(
        'user_feedbacks',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('message_id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=100), nullable=False),
        sa.Column('vote', sa.String(length=10), nullable=False),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.ForeignKeyConstraint(['message_id'], ['chat_messages.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_feedbacks_message_id'), 'user_feedbacks', ['message_id'], unique=False)
    op.create_index(op.f('ix_user_feedbacks_user_id'), 'user_feedbacks', ['user_id'], unique=False)

    # 4. evaluation_logs
    op.create_table(
        'evaluation_logs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('run_type', sa.String(length=50), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('answer', sa.Text(), nullable=False),
        sa.Column('context_snippet', sa.Text(), nullable=True),
        sa.Column('faithfulness', sa.Float(), nullable=True),
        sa.Column('answer_relevancy', sa.Float(), nullable=True),
        sa.Column('judge_model', sa.String(length=50), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='PASSED'),
        sa.Column('run_date', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_evaluation_logs_run_type'), 'evaluation_logs', ['run_type'], unique=False)
    op.create_index(op.f('ix_evaluation_logs_run_date'), 'evaluation_logs', ['run_date'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_evaluation_logs_run_date'), table_name='evaluation_logs')
    op.drop_index(op.f('ix_evaluation_logs_run_type'), table_name='evaluation_logs')
    op.drop_table('evaluation_logs')

    op.drop_index(op.f('ix_user_feedbacks_user_id'), table_name='user_feedbacks')
    op.drop_index(op.f('ix_user_feedbacks_message_id'), table_name='user_feedbacks')
    op.drop_table('user_feedbacks')

    op.drop_index(op.f('ix_chat_messages_created_at'), table_name='chat_messages')
    op.drop_index(op.f('ix_chat_messages_session_id'), table_name='chat_messages')
    op.drop_table('chat_messages')

    op.drop_index(op.f('ix_chat_sessions_user_id'), table_name='chat_sessions')
    op.drop_table('chat_sessions')
