"""
Unit tests for the face recognition attendance system.

Tests the core business logic: ML search algorithm, data validation,
log saving, and registration — all with mocked external dependencies.

Run with: pytest tests/ -v
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root to path so we can import face_rec
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Mock heavy dependencies BEFORE importing face_rec
# This avoids needing redis/insightface installed in the test environment.
# ---------------------------------------------------------------------------
mock_redis_module = MagicMock()
mock_redis_instance = MagicMock()
mock_redis_module.StrictRedis.return_value = mock_redis_instance
mock_redis_instance.ping.return_value = True
mock_redis_instance.hgetall.return_value = {}
mock_redis_module.RedisError = Exception  # So except redis.RedisError works
mock_redis_module.ConnectionError = ConnectionError

mock_insightface_module = MagicMock()
mock_insightface_app = MagicMock()
mock_insightface_module.app.FaceAnalysis.return_value = mock_insightface_app

# Inject mocks into sys.modules before face_rec imports them
sys.modules['redis'] = mock_redis_module
sys.modules['insightface'] = mock_insightface_module
sys.modules['insightface.app'] = mock_insightface_module.app
sys.modules['streamlit_webrtc'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['dotenv'] = MagicMock()

# Make dotenv.load_dotenv a no-op
sys.modules['dotenv'].load_dotenv = lambda: None

# Set env vars before import
os.environ['REDIS_HOST'] = 'localhost'
os.environ['REDIS_PORT'] = '6379'
os.environ['REDIS_PASSWORD'] = 'fake'

import face_rec


# ---------------------------------------------------------------------------
# Helper: build a fake "registered faces" DataFrame (same schema as Redis)
# ---------------------------------------------------------------------------
def create_test_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame mimicking data retrieved from Redis."""
    np.random.seed(42)
    embeddings = [np.random.randn(512).astype(np.float32) for _ in range(3)]
    embeddings = [e / np.linalg.norm(e) for e in embeddings]

    data = {
        'id_name_country': ['001@Alice@USA', '002@Bob@UK', '003@Charlie@India'],
        'Facial Features': embeddings,
        'ID': ['001', '002', '003'],
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Country': ['USA', 'UK', 'India'],
    }
    return pd.DataFrame(data)


# ===========================================================================
# Test: ml_search_algorithm
# ===========================================================================
class TestMLSearchAlgorithm:
    """Tests for the cosine-similarity face matching function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
        self.df = create_test_dataframe()

    def test_exact_match_returns_correct_person(self):
        """When test vector matches a known face exactly, return that person."""
        # Use Alice's exact embedding as the test vector
        alice_embedding = self.df.loc[0, 'Facial Features']
        person_id, person_name, person_country = face_rec.ml_search_algorithm(
            dataframe=self.df,
            feature_column='Facial Features',
            test_vector=alice_embedding,
            thresh=0.5
        )
        assert person_id == '001'
        assert person_name == 'Alice'
        assert person_country == 'USA'

    def test_no_match_returns_unknown(self):
        """When no face exceeds the threshold, return 'unknown'."""
        # Create a random vector very different from all stored faces
        np.random.seed(999)
        random_vector = np.random.randn(512).astype(np.float32)

        person_id, person_name, person_country = face_rec.ml_search_algorithm(
            dataframe=self.df,
            feature_column='Facial Features',
            test_vector=random_vector,
            thresh=0.99  # Very high threshold — nothing will match
        )
        assert person_id == 'unknown'
        assert person_name == 'unknown'
        assert person_country == 'unknown'

    def test_threshold_boundary(self):
        """Threshold = 0.0 should match everyone; highest cosine wins."""
        np.random.seed(123)
        random_vector = np.random.randn(512).astype(np.float32)

        person_id, person_name, person_country = face_rec.ml_search_algorithm(
            dataframe=self.df,
            feature_column='Facial Features',
            test_vector=random_vector,
            thresh=0.0  # Everything matches — best cosine wins
        )
        # Should return *someone*, not unknown
        assert person_id != 'unknown'
        assert person_name != 'unknown'

    def test_returns_tuple_of_three_strings(self):
        """Return type should always be a tuple of 3 strings."""
        np.random.seed(0)
        vec = np.random.randn(512).astype(np.float32)
        result = face_rec.ml_search_algorithm(
            dataframe=self.df,
            feature_column='Facial Features',
            test_vector=vec,
            thresh=0.5
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(r, str) for r in result)


# ===========================================================================
# Test: RealTimePred
# ===========================================================================
class TestRealTimePred:
    """Tests for the real-time prediction log management."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset the mock Redis before each test."""
        mock_redis_instance.reset_mock()

    def test_initial_logs_are_empty(self):
        """Newly created RealTimePred should have empty logs."""
        pred = face_rec.RealTimePred()
        assert pred.logs == dict(id=[], name=[], country=[], current_time=[])

    def test_reset_dict_clears_logs(self):
        """reset_dict() should empty all log lists."""
        pred = face_rec.RealTimePred()
        pred.logs['id'].append('001')
        pred.logs['name'].append('Alice')
        pred.logs['country'].append('USA')
        pred.logs['current_time'].append('2026-02-26')
        pred.reset_dict()
        assert all(len(v) == 0 for v in pred.logs.values())

    def test_saveLogs_redis_filters_unknown(self):
        """saveLogs_redis should NOT save 'unknown' entries to Redis."""
        pred = face_rec.RealTimePred()
        pred.logs['id'].extend(['unknown', '001'])
        pred.logs['name'].extend(['unknown', 'Alice'])
        pred.logs['country'].extend(['unknown', 'USA'])
        pred.logs['current_time'].extend(['2026-02-26', '2026-02-26'])

        pred.saveLogs_redis()

        # Redis lpush should be called with only the non-unknown entry
        mock_redis_instance.lpush.assert_called_once()
        call_args = mock_redis_instance.lpush.call_args
        assert len(call_args[0]) == 2  # key + 1 data entry (not the 'unknown' one)

    def test_saveLogs_redis_resets_after_save(self):
        """Logs should be cleared after saving, even on error."""
        pred = face_rec.RealTimePred()
        pred.logs['id'].append('001')
        pred.logs['name'].append('Alice')
        pred.logs['country'].append('USA')
        pred.logs['current_time'].append('2026-02-26')

        pred.saveLogs_redis()

        assert all(len(v) == 0 for v in pred.logs.values())


# ===========================================================================
# Test: RegistrationForm validation
# ===========================================================================
class TestRegistrationForm:
    """Tests for the registration form input validation."""

    def test_empty_id_returns_error(self):
        """Empty ID should be rejected."""
        form = face_rec.RegistrationForm()
        result = form.save_data_in_redis_db('', 'Alice', 'USA')
        assert result == 'id_name_false'

    def test_empty_name_returns_error(self):
        """Empty name should be rejected."""
        form = face_rec.RegistrationForm()
        result = form.save_data_in_redis_db('001', '', 'USA')
        assert result == 'id_name_false'

    def test_spaces_only_id_returns_error(self):
        """ID with only spaces should be rejected."""
        form = face_rec.RegistrationForm()
        result = form.save_data_in_redis_db('   ', 'Alice', 'USA')
        assert result == 'id_name_false'

    def test_none_id_returns_error(self):
        """None ID should be rejected."""
        form = face_rec.RegistrationForm()
        result = form.save_data_in_redis_db(None, 'Alice', 'USA')
        assert result == 'id_name_false'

    def test_none_name_returns_error(self):
        """None name should be rejected."""
        form = face_rec.RegistrationForm()
        result = form.save_data_in_redis_db('001', None, 'USA')
        assert result == 'id_name_false'

    def test_missing_embedding_file_returns_error(self):
        """If face_embedding.txt doesn't exist, return 'file_false'."""
        form = face_rec.RegistrationForm()
        # Ensure the file does NOT exist
        if os.path.exists('face_embedding.txt'):
            os.remove('face_embedding.txt')
        result = form.save_data_in_redis_db('001', 'Alice', 'USA')
        assert result == 'file_false'

    def test_initial_sample_count_is_zero(self):
        """New RegistrationForm should start with 0 samples."""
        form = face_rec.RegistrationForm()
        assert form.sample == 0

    def test_reset_sets_sample_to_zero(self):
        """reset() should set sample back to 0."""
        form = face_rec.RegistrationForm()
        form.sample = 10
        form.reset()
        assert form.sample == 0
