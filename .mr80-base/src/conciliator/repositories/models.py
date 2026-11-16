from datetime import UTC, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, JSON, MEDIUMTEXT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class MeegleAINodeTask(Base):
    __tablename__ = "meegle_ai_node_tasks"
    __table_args__ = (
        UniqueConstraint("task_id", name="uk_meegle_ai_tasks_task_id"),
        Index(
            "idx_meegle_ai_tasks_state_node",
            "project_key",
            "work_item_id",
            "state_key",
            "id",
        ),
        Index(
            "idx_meegle_ai_tasks_uuid_node",
            "project_key",
            "work_item_id",
            "node_uuid",
            "id",
        ),
        Index("idx_meegle_ai_tasks_status_updated", "status", "updated_at"),
        {"comment": "Meegle AI节点任务"},
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True).with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="主键ID",
    )
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="Webhook幂等任务ID")
    project_key: Mapped[str] = mapped_column(String(128), nullable=False, comment="空间Key")
    work_item_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="工作项ID")
    work_item_type_key: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="工作项类型Key")
    state_key: Mapped[str] = mapped_column(String(128), nullable=False, comment="节点状态Key")
    node_uuid: Mapped[str] = mapped_column(String(128), nullable=False, comment="AI节点UUID")
    user_key: Mapped[str] = mapped_column(String(128), nullable=False, comment="触发用户Key")
    props: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list, comment="AI节点属性")
    action_token: Mapped[str] = mapped_column(String(128), nullable=False, comment="卡片操作令牌")
    status: Mapped[str] = mapped_column(String(32), nullable=False, comment="任务状态")
    lease_owner: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    timing: Mapped[dict | None] = mapped_column(JSON, nullable=True, comment="自动处理与人工等待耗时")
    requirement_url: Mapped[str] = mapped_column(String(1024), nullable=False, default="", comment="需求文档链接")
    requirement_text: Mapped[str | None] = mapped_column(
        Text().with_variant(MEDIUMTEXT(), "mysql"), nullable=True, comment="需求文档正文"
    )
    evaluation: Mapped[dict | None] = mapped_column(JSON, nullable=True, comment="输入评估结果")
    evaluation_rounds: Mapped[list[dict] | None] = mapped_column(
        JSON, nullable=True, default=list, comment="输入评估轮次与用户反馈"
    )
    repo_id: Mapped[str] = mapped_column(String(128), nullable=False, default="", comment="代码仓库ID")
    work_branch: Mapped[str] = mapped_column(String(255), nullable=False, default="", comment="编码分支")
    coco_session_id: Mapped[str] = mapped_column(String(128), nullable=False, default="", comment="Coco Session ID")
    coco_run_id: Mapped[str] = mapped_column(String(128), nullable=False, default="", comment="Coco Run ID")
    coco_last_event_id: Mapped[str] = mapped_column(String(128), nullable=False, default="", comment="Coco最后事件ID")
    mr_id: Mapped[str] = mapped_column(String(128), nullable=False, default="", comment="代码评审ID")
    mr_iid: Mapped[int | None] = mapped_column(BigInteger, nullable=True, comment="代码评审IID")
    mr_url: Mapped[str] = mapped_column(String(1024), nullable=False, default="", comment="代码评审链接")
    platform_status_done: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0", comment="Meegle状态是否已完成"
    )
    completion_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0")
    claimed_request_ids: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, comment="已处理的卡片请求ID"
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True, comment="最后错误")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow, comment="更新时间"
    )


class MeegleAINodeRound(Base):
    __tablename__ = "meegle_ai_node_rounds"
    __table_args__ = (
        UniqueConstraint("task_id", "round_no", name="uk_meegle_ai_rounds_task_no"),
        Index("idx_meegle_ai_rounds_task", "task_id", "round_no"),
        {"comment": "Meegle AI节点编码交互轮次"},
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True).with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="主键ID",
    )
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="关联任务ID")
    round_no: Mapped[int] = mapped_column(Integer, nullable=False, comment="轮次序号")
    status: Mapped[str] = mapped_column(String(32), nullable=False, comment="轮次状态")
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True, comment="用户反馈")
    trigger_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="initial", comment="轮次触发类型"
    )
    reviewed_round_no: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="反馈针对的轮次"
    )
    base_commit_sha: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", comment="本轮起始提交SHA"
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True, comment="本轮摘要")
    commit_sha: Mapped[str] = mapped_column(String(64), nullable=False, default="", comment="提交SHA")
    code_url: Mapped[str] = mapped_column(String(1024), nullable=False, default="", comment="代码链接")
    tests: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list, comment="测试记录")
    unresolved_items: Mapped[list[str] | None] = mapped_column(JSON, nullable=True, comment="编码阻塞或未完成项")
    failure_code: Mapped[str | None] = mapped_column(Text, nullable=True, default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True, comment="本轮错误")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, comment="开始时间")
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="结束时间")


class MeegleAINodePendingFeedback(Base):
    """Feedback acknowledged to a reviewer before a coding round can start.

    The webhook reply is sent as soon as a comment is accepted, so the text has
    to outlive the process until the aggregation window closes.
    """

    __tablename__ = "meegle_ai_node_pending_feedback"
    __table_args__ = (
        UniqueConstraint(
            "task_id", "comment_id", name="uk_meegle_ai_pending_task_comment"
        ),
        Index("idx_meegle_ai_pending_task", "task_id", "available_at"),
        {"comment": "Meegle AI节点待处理评论反馈"},
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True).with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="主键ID",
    )
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="关联任务ID")
    comment_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="平台评论ID")
    feedback: Mapped[str] = mapped_column(Text, nullable=False, comment="待提交反馈")
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, comment="聚合窗口结束时间"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, comment="创建时间"
    )


class MeegleAINodeCommentClaim(Base):
    """A comment a worker has taken responsibility for answering.

    The unique key is the cross-worker deduplication point; ``expires_at``
    bounds how long a crashed worker can keep a question unanswered.
    """

    __tablename__ = "meegle_ai_node_comment_claim"
    __table_args__ = (
        UniqueConstraint(
            "task_id", "request_id", name="uk_meegle_ai_claim_task_request"
        ),
        Index("idx_meegle_ai_claim_expiry", "expires_at"),
        {"comment": "Meegle AI节点评论处理声明"},
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True).with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="主键ID",
    )
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="关联任务ID")
    request_id: Mapped[str] = mapped_column(
        String(160), nullable=False, comment="评论请求ID"
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, comment="声明过期时间"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, comment="创建时间"
    )


class MeegleAINodeQuestionRecord(Base):
    """Durable record of a review question that must be answered.

    The webhook acknowledges the delivery as soon as the record is written, so
    an answer that fails afterwards (agent error, publish failure, worker
    crash) is not lost: the recovery scan picks the record up again, either to
    finish the answer or to republish a saved answer.
    """

    __tablename__ = "meegle_ai_node_question_record"
    __table_args__ = (
        UniqueConstraint(
            "task_id", "comment_id", name="uk_meegle_ai_question_task_comment"
        ),
        Index("idx_meegle_ai_question_status", "status"),
        {"comment": "Meegle AI节点评论提问记录"},
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True).with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="主键ID",
    )
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="关联任务ID")
    comment_id: Mapped[str] = mapped_column(
        String(160), nullable=False, comment="提问评论ID"
    )
    repo_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="仓库ID")
    mr_iid: Mapped[int] = mapped_column(Integer, nullable=False, comment="MR编号")
    discussion_id: Mapped[str | None] = mapped_column(
        String(160), nullable=True, comment="回复所在讨论ID"
    )
    position: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, comment="回复位置"
    )
    parent: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, comment="被引用的父评论"
    )
    question_text: Mapped[str] = mapped_column(
        Text, nullable=False, comment="问题正文"
    )
    author_email: Mapped[str] = mapped_column(
        String(160), nullable=False, comment="提问者邮箱"
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", comment="处理状态"
    )
    answer_text: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="已生成的回答正文"
    )
    session_id: Mapped[str | None] = mapped_column(
        String(160), nullable=True, comment="Coco会话ID"
    )
    run_id: Mapped[str | None] = mapped_column(
        String(160), nullable=True, comment="Coco问答运行ID"
    )
    event_id: Mapped[str | None] = mapped_column(
        String(160), nullable=True, comment="Coco问答事件ID"
    )
    attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="回答尝试次数"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        comment="更新时间",
    )


class MeegleAINodeAttempt(Base):
    __tablename__ = "meegle_ai_node_attempts"
    __table_args__ = (
        UniqueConstraint(
            "task_id", "round_no", "attempt_no", name="uk_meegle_ai_attempts_task_round_no"
        ),
        Index(
            "idx_meegle_ai_attempts_task_round",
            "task_id",
            "round_no",
            "attempt_no",
        ),
        {"comment": "Meegle AI节点轮次执行尝试"},
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True).with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="主键ID",
    )
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="关联任务ID")
    round_no: Mapped[int] = mapped_column(Integer, nullable=False, comment="业务轮次序号")
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False, comment="轮内执行序号")
    status: Mapped[str] = mapped_column(String(32), nullable=False, comment="执行状态")
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, default="", comment="Coco Session ID")
    run_id: Mapped[str] = mapped_column(String(128), nullable=False, default="", comment="Coco Run ID")
    event_id: Mapped[str] = mapped_column(String(128), nullable=False, default="", comment="Coco起始事件ID")
    error: Mapped[str | None] = mapped_column(Text, nullable=True, comment="执行错误")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, comment="开始时间")
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="结束时间")


class ResolveTask(Base):
    __tablename__ = "resolve_tasks"
    __table_args__ = (
        Index("idx_tasks_repo_mr_status", "repo_id", "original_mr_id", "status"),
        Index("idx_tasks_repo_new_mr_status", "repo_id", "new_mr_id", "status"),
        Index("idx_tasks_status_updated", "status", "updated_at"),
        UniqueConstraint("active_key", name="uk_tasks_active_key"),
        {"comment": "解冲突任务主表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger(),
        primary_key=True,
        autoincrement=True,
        comment="主键ID",
    )
    repo_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="仓库ID")
    source_branch: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="MR源分支",
    )
    target_branch: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="MR目标分支",
    )
    resolve_branch: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="解冲突分支",
    )
    original_mr_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="原始MR的IID",
    )
    original_mr_url: Mapped[str | None] = mapped_column(String(512))
    trace_schema_version: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    active_key: Mapped[str | None] = mapped_column(
        String(191),
        nullable=True,
        comment="非终态处理版本唯一键",
    )
    conflict_files_snapshot: Mapped[list[dict] | None] = mapped_column(
        JSON, nullable=True, comment="触发时冻结的冲突文件内容"
    )
    mr_version: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="任务冻结的MR Version Number",
    )
    source_commit_sha: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="任务冻结的MR源分支commit",
    )
    target_commit_sha: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="任务冻结的MR目标分支commit",
    )
    resolve_base_sha: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="resolve分支创建基线commit",
    )
    resolve_commit_sha: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="Coco推送且服务端验收通过的commit",
    )
    trigger_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="触发类型(COMMENT/AUTO_CONFLICT)",
    )
    trigger_payload: Mapped[dict] = mapped_column(JSON, nullable=False, comment="触发webhook原始payload")
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="PENDING",
        comment="任务状态",
    )
    coco_task_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="兼容字段，值为Coco Session ID",
    )
    coco_workspace_id: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="Coco Workspace ID")
    coco_agent_id: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="Coco Agent ID")
    coco_session_id: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="Coco Session ID")
    coco_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="Coco当前Run ID")
    coco_last_event_id: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="Coco最后消费的Event ID")
    coco_result_schema_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="Coco结构化结果schema版本"
    )
    coco_result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True, comment="通过schema校验的Coco结构化结果")
    lark_doc_url: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="飞书文档URL",
    )
    new_mr_id: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        comment="解冲突新MR的IID",
    )
    new_mr_url: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="NextCode回读的解冲突新MR URL",
    )
    mr_author_email: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="原始MR作者邮箱",
    )
    trigger_author_email: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="COMMENT触发时输入/resolve的用户邮箱",
    )
    last_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="最后错误信息",
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="重试次数",
    )
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="乐观锁版本号",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        comment="更新时间",
    )

    blocks: Mapped[list["ConflictBlock"]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        primaryjoin="ResolveTask.id == foreign(ConflictBlock.task_id)",
    )
    events: Mapped[list["TaskEvent"]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        primaryjoin="ResolveTask.id == foreign(TaskEvent.task_id)",
    )


class ConflictBlock(Base):
    __tablename__ = "conflict_blocks"
    __table_args__ = (
        Index("idx_conflict_blocks_task_id", "task_id"),
        UniqueConstraint("task_id", "block_idx", name="uk_blocks_task_idx"),
        UniqueConstraint("task_id", "conflict_id", name="uk_blocks_task_conflict"),
        UniqueConstraint("discussion_id", name="uk_blocks_discussion"),
        {"comment": "冲突块详情"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger(),
        primary_key=True,
        autoincrement=True,
        comment="主键ID",
    )
    task_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="关联resolve_tasks.id",
    )
    conflict_id: Mapped[str] = mapped_column(String(32), nullable=False, comment="任务内稳定冲突块ID")
    block_idx: Mapped[int] = mapped_column(Integer, nullable=False, comment="块序号(0-based)")
    file_path: Mapped[str] = mapped_column(String(512), nullable=False, comment="冲突文件路径")
    start_line: Mapped[int] = mapped_column(Integer, nullable=False, comment="起始行号")
    end_line: Mapped[int] = mapped_column(Integer, nullable=False, comment="结束行号")
    ours_text: Mapped[str] = mapped_column(Text, nullable=True, comment="源分支冲突内容")
    theirs_text: Mapped[str] = mapped_column(Text, nullable=True, comment="目标分支冲突内容")
    ai_resolution: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="AI解冲突方案",
    )
    author_email: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="兼容字段，优先保存源侧author邮箱",
    )
    source_author_email: Mapped[str | None] = mapped_column(
        String(255), nullable=True, comment="经服务端复核的源侧author邮箱"
    )
    source_author_commit_sha: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="源侧最后修改commit"
    )
    source_path: Mapped[str | None] = mapped_column(String(512), nullable=True, comment="源侧归因文件路径")
    source_start_line: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="源侧归因起始行")
    source_end_line: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="源侧归因结束行")
    target_author_email: Mapped[str | None] = mapped_column(
        String(255), nullable=True, comment="经服务端复核的目标侧author邮箱"
    )
    target_author_commit_sha: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="目标侧最后修改commit"
    )
    target_path: Mapped[str | None] = mapped_column(String(512), nullable=True, comment="目标侧归因文件路径")
    target_start_line: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="目标侧归因起始行")
    target_end_line: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="目标侧归因结束行")
    attribution_status: Mapped[str | None] = mapped_column(String(32), nullable=True, comment="归因复核状态")
    decision_summary: Mapped[str | None] = mapped_column(Text, nullable=True, comment="Agent给出的决策摘要")
    introduced_fix: Mapped[str | None] = mapped_column(Text, nullable=True, comment="组合两侧后额外引入的修复")
    resolved_path: Mapped[str | None] = mapped_column(
        String(512), nullable=True, comment="resolve commit中的文件路径"
    )
    resolved_start_line: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="resolve commit中的起始行"
    )
    resolved_end_line: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="resolve commit中的结束行"
    )
    discussion_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="MR Discussion ID",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="创建时间",
    )

    task: Mapped["ResolveTask"] = relationship(
        back_populates="blocks",
        primaryjoin="foreign(ConflictBlock.task_id) == ResolveTask.id",
    )
    approval: Mapped["Approval | None"] = relationship(
        back_populates="block",
        uselist=False,
        cascade="all, delete-orphan",
        primaryjoin="foreign(Approval.block_id) == ConflictBlock.id",
    )


class Approval(Base):
    __tablename__ = "approvals"
    __table_args__ = (
        UniqueConstraint("block_id", name="uk_approvals_block_id"),
        {"comment": "逐块审批状态"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger(),
        primary_key=True,
        autoincrement=True,
        comment="主键ID",
    )
    block_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="关联conflict_blocks.id",
    )
    reviewer_email: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="审批人邮箱",
    )
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="PENDING",
        comment="审批状态(PENDING/APPROVED/REJECTED/NOT_REQUIRED)",
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="审批时间",
    )
    comment: Mapped[str | None] = mapped_column(Text, nullable=True, comment="审批备注(reject原因)")
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="乐观锁版本号",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        comment="更新时间",
    )

    block: Mapped["ConflictBlock"] = relationship(
        back_populates="approval",
        primaryjoin="foreign(Approval.block_id) == ConflictBlock.id",
    )


class TaskEvent(Base):
    __tablename__ = "task_events"
    __table_args__ = (
        Index("idx_task_events_task_id", "task_id"),
        UniqueConstraint("task_id", "coco_event_id", name="uk_events_coco"),
        Index("idx_events_task_time", "task_id", "occurred_at"),
        {"comment": "任务事件审计日志"},
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True).with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="主键ID",
    )
    task_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="关联resolve_tasks.id",
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, comment="事件类型")
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, comment="事件原始payload")
    coco_event_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Coco事件ID(去重用)",
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="事件发生时间",
    )

    task: Mapped["ResolveTask"] = relationship(
        back_populates="events",
        primaryjoin="foreign(TaskEvent.task_id) == ResolveTask.id",
    )


class NextCodeWebhookDelivery(Base):
    __tablename__ = "nextcode_webhook_deliveries"
    __table_args__ = (
        UniqueConstraint("delivery_key", name="uk_nextcode_delivery_key"),
        Index("idx_nextcode_delivery_event", "event_type", "created_at"),
        {"comment": "NextCode Webhook投递去重记录"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger(),
        primary_key=True,
        autoincrement=True,
        comment="主键ID",
    )
    delivery_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Delivery/Event去重键",
    )
    event_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="X-Vecode-Event-Id",
    )
    delivery_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="X-Vecode-Delivery",
    )
    event_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="X-Vecode-Event",
    )
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="PROCESSING",
        comment="处理状态",
    )
    processed_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="处理条目数",
    )
    last_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="最后错误",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        comment="更新时间",
    )


class LarkNotificationOutbox(Base):
    __tablename__ = "lark_notification_outbox"
    __table_args__ = (
        UniqueConstraint(
            "task_id",
            "notification_type",
            name="uk_lark_notification_task_type",
        ),
        UniqueConstraint(
            "request_uuid",
            name="uk_lark_notification_request_uuid",
        ),
        Index(
            "idx_lark_notification_status_due",
            "status",
            "next_retry_at",
            "updated_at",
        ),
        {"comment": "飞书通知可靠投递箱"},
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True).with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="主键ID",
    )
    task_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="关联resolve_tasks.id",
    )
    notification_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="通知类型",
    )
    recipient_email: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="收件人邮箱",
    )
    recipient_open_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="邮箱解析得到的飞书open_id",
    )
    mr_url: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="通知中的解冲突MR URL",
    )
    request_uuid: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="飞书消息发送幂等键",
    )
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="PENDING",
        comment="投递状态(PENDING/SENDING/RETRY/SENT/FAILED)",
    )
    attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="已领取投递次数",
    )
    next_retry_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="下次可重试时间",
    )
    message_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="飞书发送成功返回的message_id",
    )
    last_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="最后投递错误",
    )
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="乐观锁版本号",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        comment="更新时间",
    )


class RepoConfig(Base):
    __tablename__ = "repo_configs"
    __table_args__ = (
        UniqueConstraint("repo_id", name="uk_repo_configs_repo_id"),
        {"comment": "仓库配置(自动解冲突开关等)"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger(),
        primary_key=True,
        autoincrement=True,
        comment="主键ID",
    )
    repo_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="仓库ID")
    auto_resolve_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="是否开启自动解冲突",
    )
    allowed_target_branches: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="允许自动解冲突的目标分支白名单",
    )
    agent_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Coco Agent ID",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        comment="更新时间",
    )


class TaskTraceEvent(Base):
    __tablename__ = "task_trace_events"
    __table_args__ = (
        UniqueConstraint("event_uid", name="uk_trace_uid"),
        UniqueConstraint("source_task_event_id", name="uk_trace_business_event"),
        Index("idx_trace_time", "occurred_at", "id"),
        Index("idx_trace_task_time", "task_id", "occurred_at", "id"),
        Index("idx_trace_call", "call_id", "id"),
        Index("idx_trace_context", "trace_id", "id"),
        Index("idx_trace_type_time", "event_type", "occurred_at", "id"),
    )
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True).with_variant(Integer, "sqlite"), primary_key=True)
    event_uid: Mapped[str] = mapped_column(String(64), nullable=False)
    source_task_event_id: Mapped[int | None] = mapped_column(BigInteger)
    task_id: Mapped[int | None] = mapped_column(BigInteger)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime().with_variant(DATETIME(fsp=6), "mysql"), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    task_status: Mapped[str | None] = mapped_column(String(32))
    from_status: Mapped[str | None] = mapped_column(String(32))
    to_status: Mapped[str | None] = mapped_column(String(32))
    task_version: Mapped[int | None] = mapped_column(Integer)
    provider: Mapped[str | None] = mapped_column(String(16))
    operation: Mapped[str | None] = mapped_column(String(128))
    call_id: Mapped[str | None] = mapped_column(String(64))
    operation_id: Mapped[str | None] = mapped_column(String(64))
    attempt: Mapped[int | None] = mapped_column(Integer)
    execution_id: Mapped[str | None] = mapped_column(String(64))
    retry_round: Mapped[int | None] = mapped_column(Integer)
    outcome: Mapped[str | None] = mapped_column(String(32))
    http_status: Mapped[int | None] = mapped_column(Integer)
    duration_ms: Mapped[int | None] = mapped_column(BigInteger)
    external_request_id: Mapped[str | None] = mapped_column(String(128))
    summary: Mapped[str] = mapped_column(String(1024), default="")
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    body_state: Mapped[str] = mapped_column(String(32), default="EMPTY")
    body_bytes: Mapped[int] = mapped_column(BigInteger, default=0)


class TaskTraceBody(Base):
    __tablename__ = "task_trace_event_bodies"
    __table_args__ = (UniqueConstraint("event_id", "body_kind", "chunk_index", name="uk_trace_body_chunk"),)
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True).with_variant(Integer, "sqlite"), primary_key=True)
    event_id: Mapped[int] = mapped_column(BIGINT(unsigned=True).with_variant(Integer, "sqlite"), nullable=False)
    body_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str | None] = mapped_column(Text().with_variant(MEDIUMTEXT(), "mysql"))
    byte_count: Mapped[int] = mapped_column(Integer)


class TaskTraceLink(Base):
    __tablename__ = "task_trace_links"
    __table_args__ = (
        UniqueConstraint("trace_id", "task_id", name="uk_trace_link"),
        Index("idx_trace_link_task", "task_id", "trace_id"),
    )
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True).with_variant(Integer, "sqlite"), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    task_id: Mapped[int] = mapped_column(BigInteger, nullable=False)


class DeploymentWorker(Base):
    __tablename__ = "aeolus_deployment_workers"

    __table_args__ = (UniqueConstraint("worker_id", name="uk_aeolus_worker_id"),)

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True).with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True, comment="主键ID"
    )
    worker_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # Preserve business-key Session.get() lookups independently of the physical primary key.
    __mapper_args__ = {"primary_key": [worker_id]}
    vmid: Mapped[str | None] = mapped_column(String(128), nullable=True, unique=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ready: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)


class Deployment(Base):
    __tablename__ = "aeolus_deployments"
    __table_args__ = (
        UniqueConstraint("task_id", "request_id", name="uk_aeolus_deploy_request"),
        Index("idx_aeolus_deploy_worker_status", "worker_id", "status", "created_at"),
        Index("idx_aeolus_deploy_task", "task_id", "created_at"),
        UniqueConstraint("deployment_id", name="uk_aeolus_deployment_id"),
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True).with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True, comment="主键ID"
    )
    deployment_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # Preserve business-key Session.get() lookups independently of the physical primary key.
    __mapper_args__ = {"primary_key": [deployment_id]}
    task_id: Mapped[str] = mapped_column(String(128), nullable=False)
    worker_id: Mapped[str] = mapped_column(String(64), nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    round_no: Mapped[int] = mapped_column(Integer, nullable=False)
    commit_sha: Mapped[str] = mapped_column(String(40), nullable=False)
    generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False, default="dispatching")
    created_at: Mapped[datetime] = mapped_column(DateTime().with_variant(DATETIME(fsp=6), "mysql"), nullable=False)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)


class LarkUserToken(Base):
    """Per-user Feishu OAuth token, used to read requirement documents as the operator.

    A tenant token cannot read a document that was never shared with the app, so
    documents are read with the identity of the user who triggered the task. That
    user authorizes once; the refresh token is stored here and rotated on use.
    """

    __tablename__ = "lark_user_tokens"
    __table_args__ = (
        UniqueConstraint("open_id", name="uk_lark_user_tokens_open_id"),
        UniqueConstraint("user_key", name="uk_lark_user_tokens_user_key"),
        Index("idx_lark_user_tokens_user_key", "user_key"),
        {"comment": "飞书用户身份访问令牌"},
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True).with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="主键ID",
    )
    open_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="飞书用户 open_id")
    user_key: Mapped[str] = mapped_column(
        String(128), nullable=False, default="", comment="最近一次授权对应的 Meegle user_key"
    )
    refresh_token: Mapped[str] = mapped_column(Text, nullable=False, comment="刷新令牌")
    access_token: Mapped[str] = mapped_column(
        Text, nullable=False, default="", comment="访问令牌缓存"
    )
    access_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="访问令牌过期时间"
    )
    refresh_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="刷新令牌过期时间"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True).with_variant(DATETIME(fsp=6), "mysql"),
        nullable=False, default=_utcnow, comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True).with_variant(DATETIME(fsp=6), "mysql"),
        nullable=False, default=_utcnow, onupdate=_utcnow, comment="更新时间",
    )


# Register the independent PPE tables in the shared metadata used by migrations and tests.
from conciliator.repositories import ppe_models as _ppe_models  # noqa: E402, F401
