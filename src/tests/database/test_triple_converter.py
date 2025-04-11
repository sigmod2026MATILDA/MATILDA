# # tests/database/test_triple_converter.py
#
# import pytest
# import logging
# import tempfile
# import os
# from sqlalchemy import (
#     MetaData,
#     Table,
#     Column,
#     Integer,
#     String,
#     ForeignKey,
#     create_engine,
# )
# from database.triple_converter import TripleConverter  # Updated import path
#
#
# @pytest.fixture(scope='module')
# def temp_db_file():
#     """
#     Creates a temporary file-based SQLite database.
#     """
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
#         db_path = tmp.name
#     yield db_path
#     os.remove(db_path)
#
#
# @pytest.fixture(scope='module')
# def engine(temp_db_file):
#     """
#     Creates a SQLite engine connected to the temporary file-based database.
#     """
#     engine = create_engine(
#         f'sqlite:///{temp_db_file}',
#         echo=False,
#         connect_args={
#             'check_same_thread': False
#         }
#     )
#     return engine
#
#
# @pytest.fixture(scope='module')
# def metadata():
#     """
#     Provides a MetaData instance for table definitions.
#     """
#     return MetaData()
#
#
# @pytest.fixture(scope='module')
# def setup_database(engine, metadata):
#     """
#     Sets up the database schema and inserts sample data.
#     """
#     # Define tables
#     users = Table('users', metadata,
#                   Column('id', Integer, primary_key=True),
#                   Column('name', String, nullable=False),
#                   Column('email', String, nullable=False))
#
#     posts = Table('posts', metadata,
#                   Column('id', Integer, primary_key=True),
#                   Column('user_id', Integer, ForeignKey('users.id'), nullable=False),
#                   Column('title', String, nullable=False),
#                   Column('content', String, nullable=False))
#
#     comments = Table('comments', metadata,
#                      Column('id', Integer, primary_key=True),
#                      Column('post_id', Integer, ForeignKey('posts.id'), nullable=False),
#                      Column('user_id', Integer, ForeignKey('users.id'), nullable=False),
#                      Column('comment', String, nullable=False))
#
#     # Create tables
#     metadata.create_all(engine)
#
#     # Insert sample data
#     with engine.connect() as conn:
#         conn.execute(users.insert(), [
#             {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'},
#             {'id': 2, 'name': 'Bob', 'email': 'bob@example.com'}
#         ])
#         conn.execute(posts.insert(), [
#             {'id': 1, 'user_id': 1, 'title': 'Post 1', 'content': 'Content 1'},
#             {'id': 2, 'user_id': 1, 'title': 'Post 2', 'content': 'Content 2'},
#             {'id': 3, 'user_id': 2, 'title': 'Post 3', 'content': 'Content 3'},
#         ])
#         conn.execute(comments.insert(), [
#             {'id': 1, 'post_id': 1, 'user_id': 2, 'comment': 'Comment 1'},
#             {'id': 2, 'post_id': 2, 'user_id': 1, 'comment': 'Comment 2'},
#             {'id': 3, 'post_id': 3, 'user_id': 1, 'comment': 'Comment 3'},
#         ])
#
#
# def test_triple_converter(engine, metadata, setup_database):
#     """
#     Tests the TripleConverter's convert_to_triples method.
#     """
#     # Setup logging
#     logger = logging.getLogger('TripleConverterTest')
#     logger.setLevel(logging.DEBUG)
#     # To avoid duplicate logs in pytest output, check if handlers are already present
#     if not logger.handlers:
#         logger.addHandler(logging.StreamHandler())
#
#     # Initialize TripleConverter
#     converter = TripleConverter(engine, metadata, logger)
#
#     # Convert to triples
#     triples = converter.convert_to_triples()
#
#     # Expected triples
#     expected_triples = [
#         # Users table literals
#         ('users_1', 'users.name', '"Alice"'),
#         ('users_1', 'users.email', '"alice@example.com"'),
#         ('users_2', 'users.name', '"Bob"'),
#         ('users_2', 'users.email', '"bob@example.com"'),
#         # Posts table foreign keys and literals
#         ('posts_1', 'posts.user_id', 'users_1'),
#         ('posts_1', 'posts.title', '"Post 1"'),
#         ('posts_1', 'posts.content', '"Content 1"'),
#         ('posts_2', 'posts.user_id', 'users_1'),
#         ('posts_2', 'posts.title', '"Post 2"'),
#         ('posts_2', 'posts.content', '"Content 2"'),
#         ('posts_3', 'posts.user_id', 'users_2'),
#         ('posts_3', 'posts.title', '"Post 3"'),
#         ('posts_3', 'posts.content', '"Content 3"'),
#         # Comments table foreign keys and literals
#         ('comments_1', 'comments.post_id', 'posts_1'),
#         ('comments_1', 'comments.user_id', 'users_2'),
#         ('comments_1', 'comments.comment', '"Comment 1"'),
#         ('comments_2', 'comments.post_id', 'posts_2'),
#         ('comments_2', 'comments.user_id', 'users_1'),
#         ('comments_2', 'comments.comment', '"Comment 2"'),
#         ('comments_3', 'comments.post_id', 'posts_3'),
#         ('comments_3', 'comments.user_id', 'users_1'),
#         ('comments_3', 'comments.comment', '"Comment 3"'),
#     ]
#
#     # Assert that the generated triples match the expected triples
#     assert set(triples) == set(expected_triples), "Generated triples do not match expected triples."
