from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_redis import RedisChatMessageHistory

from agents.common.data import Message
from agents.memory.conversation_history import ConversationMessage, QueryType
from services.conversation import ConversationService

TIME_STAMP = 1.8
QUESTIONS = ["question1?", "question2?", "question3?"]
CONVERSATION_ID = "1"
POD_YAML = """
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: my-container
    image: nginx:latest
    ports:
    - containerPort: 80
"""
TEST_MESSAGE = Message(
    query="test query",
    resource_kind="Pod",
    resource_api_version="v1",
    resource_name="my-pod",
    namespace="default",
)

from threading import Thread

import redis
from fakeredis import TcpFakeServer

REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
server_address = (REDIS_HOST, REDIS_PORT)
server = TcpFakeServer(server_address)
t = Thread(target=server.serve_forever, daemon=True)
t.start()


@pytest.mark.asyncio(scope="class")
class TestConversation:
    @pytest.fixture
    def mock_model_factory(self):
        mock_model = Mock()
        with patch("services.conversation.ModelFactory") as mock:
            mock.return_value.create_model.return_value = mock_model
            yield mock

    @pytest.fixture
    def mock_kyma_graph(self):
        mock_kyma_graph = MagicMock()
        mock_kyma_graph.astream.return_value = AsyncMock()
        mock_kyma_graph.astream.return_value.__aiter__.return_value = [
            "chunk1",
            "chunk2",
            "chunk3",
        ]
        with patch(
            "services.conversation.KymaGraph", return_value=mock_kyma_graph
        ) as mock:
            yield mock

    @pytest.fixture
    def mock_redis_saver(self):
        async def async_mock_add_conversation_message(*args, **kwargs):
            pass

        with patch("services.conversation.RedisSaver") as mock:
            mock.return_value.add_conversation_message = AsyncMock(
                side_effect=async_mock_add_conversation_message
            )
            yield mock

    @pytest.fixture
    def mock_init_pool(self):
        with patch("services.conversation.initialize_async_pool") as mock:
            yield mock

    @pytest.fixture
    def mock_redis_history(self):
        mock_history = Mock()
        mock_history.add_message = Mock(return_value=None)
        with patch(
            "services.conversation.RedisChatMessageHistory", return_value=mock_history
        ):
            yield mock_history

    @pytest.mark.asyncio
    async def test_new_conversation(
        self,
        mock_model_factory,
        mock_kyma_graph,
        mock_redis_history,
    ) -> None:
        # Given:
        mock_handler = Mock()
        mock_handler.fetch_relevant_data_from_k8s_cluster = Mock(return_value=POD_YAML)
        mock_handler.generate_questions = Mock(return_value=QUESTIONS)
        conversation_service = ConversationService(
            initial_questions_handler=mock_handler
        )

        mock_k8s_client = Mock()

        # When:
        result = await conversation_service.new_conversation(
            session_id=CONVERSATION_ID, k8s_client=mock_k8s_client, message=TEST_MESSAGE
        )

        # print messages from redis
        messages = await conversation_service.redis_saver.get_all_conversation_messages(
            CONVERSATION_ID
        )
        conversation = ConversationMessage(**messages[0])

        assert conversation == ConversationMessage(
            type=QueryType.USER_QUERY,
            query=TEST_MESSAGE.query,
            response="",
            timestamp=0.0,
        )

        # Then:
        assert result == QUESTIONS

    @pytest.mark.asyncio
    async def test_handle_request(
        self, mock_model_factory, mock_init_pool, mock_redis_saver, mock_kyma_graph
    ):
        # Given:
        mock_k8s_client = Mock()

        # When:
        messaging_service = ConversationService()

        # Then:
        result = [
            chunk
            async for chunk in messaging_service.handle_request(
                CONVERSATION_ID, TEST_MESSAGE, mock_k8s_client
            )
        ]
        assert result == [b"chunk1", b"chunk2", b"chunk3"]
