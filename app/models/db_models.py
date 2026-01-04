from sqlalchemy.orm import DeclarativeBase
import sqlalchemy as sqla
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector


class IngestBase(DeclarativeBase):
    schema = "ingester"
    __table_args__ = {"schema": schema}


class ParentDocs(IngestBase):
    __tablename__ = "parent_docs"

    parent_doc_id = sqla.Column(sqla.String(36), primary_key=True)
    created_at = sqla.Column(sqla.TIMESTAMP(timezone=True), server_default=sqla.func.now())
    doc_title = sqla.Column(sqla.String, nullable=False)
    token_len = sqla.Column(sqla.Integer, nullable=True)
    file_ext = sqla.Column(sqla.String, nullable=True)

class ChildDocs(IngestBase):
    __tablename__ = "child_docs"

    child_doc_id = sqla.Column(sqla.String(36), primary_key=True)
    created_at = sqla.Column(sqla.TIMESTAMP(timezone=True), server_default=sqla.func.now())
    page_num = sqla.Column(sqla.Integer, nullable=True)
    content = sqla.Column(sqla.String, nullable=False)
    embedding = sqla.Column(Vector(1536), nullable=True)
    parent_doc_id = sqla.Column(sqla.String(36), sqla.ForeignKey("parent_docs.parent_doc_id"), nullable=False)

class Dummy(IngestBase):
    __tablename__ = "dummy"

    id = sqla.Column(sqla.Integer, primary_key=True)
    created_at = sqla.Column(sqla.TIMESTAMP(timezone=True), server_default=sqla.func.now())
    content = sqla.Column(sqla.String, nullable=False)
    embedding = sqla.Column(Vector(1536), nullable=True)