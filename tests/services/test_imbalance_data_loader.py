"""
Tests for imbalance data loader service.
"""

import pytest
from pathlib import Path
import pandas as pd
from io import StringIO
from unittest.mock import patch


@pytest.fixture
def temp_imbalance_dir(tmp_path):
    """Create temporary directory for imbalance CSV files."""
    imb_dir = tmp_path / "imbalance_data"
    imb_dir.mkdir()
    return imb_dir


@pytest.fixture
def sample_imbalance_csv_content():
    """Sample CSV content in TenneT format."""
    return """Timeinterval Start Loc;Price Shortage;Price Surplus;Regulation State
2024-10-01 00:00:00;50.5;40.5;1
2024-10-01 01:00:00;51.0;41.0;1
2024-10-01 02:00:00;52.5;42.5;-1
"""


@pytest.fixture
def sample_imbalance_csv_with_text_states():
    """Sample CSV with text-based regulation states."""
    return """Timeinterval Start Loc;Price Shortage;Price Surplus;Regulation State
2024-10-01 00:00:00;50.5;40.5;UP
2024-10-01 01:00:00;51.0;41.0;DOWN
2024-10-01 02:00:00;52.5;42.5;BALANCED
"""


class TestLoadAndCombineImbalanceCSVs:
    """Tests for load_and_combine_imbalance_csvs function."""
    
    def test_load_single_csv(self, temp_imbalance_dir, sample_imbalance_csv_content):
        """Test loading a single CSV file."""
        from app.services.imbalance_data_loader import load_and_combine_imbalance_csvs
        
        # Create CSV file
        csv_file = temp_imbalance_dir / "imbalance_2024-10.csv"
        csv_file.write_text(sample_imbalance_csv_content)
        
        result = load_and_combine_imbalance_csvs(temp_imbalance_dir)
        
        assert len(result) == 3
        assert list(result.columns) == [
            "timestamp_utc",
            "shortage_price",
            "surplus_price",
            "regulation_state"
        ]
        assert result["shortage_price"].iloc[0] == 50.5
        assert result["surplus_price"].iloc[0] == 40.5
        assert result["regulation_state"].iloc[0] == 1
    
    def test_load_multiple_csvs(self, temp_imbalance_dir):
        """Test loading and combining multiple CSV files."""
        from app.services.imbalance_data_loader import load_and_combine_imbalance_csvs
        
        # Create first CSV
        csv1_content = """Timeinterval Start Loc;Price Shortage;Price Surplus;Regulation State
2024-10-01 00:00:00;50.5;40.5;1
2024-10-01 01:00:00;51.0;41.0;1
"""
        csv1 = temp_imbalance_dir / "imbalance_oct.csv"
        csv1.write_text(csv1_content)
        
        # Create second CSV
        csv2_content = """Timeinterval Start Loc;Price Shortage;Price Surplus;Regulation State
2024-11-01 00:00:00;60.5;50.5;-1
2024-11-01 01:00:00;61.0;51.0;-1
"""
        csv2 = temp_imbalance_dir / "imbalance_nov.csv"
        csv2.write_text(csv2_content)
        
        result = load_and_combine_imbalance_csvs(temp_imbalance_dir)
        
        assert len(result) == 4
        # Verify data is sorted by timestamp
        assert result["timestamp_utc"].iloc[0] < result["timestamp_utc"].iloc[-1]
    
    def test_load_removes_duplicates(self, temp_imbalance_dir):
        """Test that duplicate timestamps are removed."""
        from app.services.imbalance_data_loader import load_and_combine_imbalance_csvs
        
        # Create CSV with duplicates
        csv_content = """Timeinterval Start Loc;Price Shortage;Price Surplus;Regulation State
2024-10-01 00:00:00;50.5;40.5;1
2024-10-01 00:00:00;55.5;45.5;1
2024-10-01 01:00:00;51.0;41.0;1
"""
        csv_file = temp_imbalance_dir / "imbalance.csv"
        csv_file.write_text(csv_content)
        
        result = load_and_combine_imbalance_csvs(temp_imbalance_dir)
        
        # Should have 2 unique timestamps (duplicate removed)
        assert len(result) == 2
        # First occurrence should be kept
        first_row = result[result["timestamp_utc"] == pd.Timestamp('2024-10-01 00:00:00')]
        assert first_row["shortage_price"].iloc[0] == 50.5
    
    def test_load_no_csv_files(self, temp_imbalance_dir):
        """Test error when no CSV files are found."""
        from app.services.imbalance_data_loader import load_and_combine_imbalance_csvs
        
        with pytest.raises(ValueError, match="No CSV files found"):
            load_and_combine_imbalance_csvs(temp_imbalance_dir)
    
    def test_load_handles_text_regulation_states(
        self, 
        temp_imbalance_dir,
        sample_imbalance_csv_with_text_states
    ):
        """Test that text-based regulation states are converted to numeric."""
        from app.services.imbalance_data_loader import load_and_combine_imbalance_csvs
        
        csv_file = temp_imbalance_dir / "imbalance.csv"
        csv_file.write_text(sample_imbalance_csv_with_text_states)
        
        result = load_and_combine_imbalance_csvs(temp_imbalance_dir)
        
        assert result["regulation_state"].iloc[0] == 1  # UP
        assert result["regulation_state"].iloc[1] == -1  # DOWN
        assert result["regulation_state"].iloc[2] == 0  # BALANCED
    
    def test_load_handles_malformed_csv(self, temp_imbalance_dir):
        """Test handling of malformed CSV file."""
        from app.services.imbalance_data_loader import load_and_combine_imbalance_csvs
        
        # Create malformed CSV (missing columns)
        bad_csv = temp_imbalance_dir / "bad.csv"
        bad_csv.write_text("timestamp;price\n2024-10-01;50\n")
        
        # Create valid CSV
        good_csv_content = """Timeinterval Start Loc;Price Shortage;Price Surplus;Regulation State
2024-10-01 00:00:00;50.5;40.5;1
"""
        good_csv = temp_imbalance_dir / "good.csv"
        good_csv.write_text(good_csv_content)
        
        # Should load the good CSV and skip the bad one
        result = load_and_combine_imbalance_csvs(temp_imbalance_dir)
        assert len(result) == 1
    
    def test_load_all_malformed_csvs(self, temp_imbalance_dir):
        """Test error when all CSV files are malformed."""
        from app.services.imbalance_data_loader import load_and_combine_imbalance_csvs
        
        bad_csv = temp_imbalance_dir / "bad.csv"
        bad_csv.write_text("invalid,content\n1,2\n")
        
        with pytest.raises(ValueError, match="Failed to load any CSV files"):
            load_and_combine_imbalance_csvs(temp_imbalance_dir)
    
    def test_load_converts_prices_to_numeric(self, temp_imbalance_dir):
        """Test that price columns are converted to numeric and missing values are forward-filled."""
        from app.services.imbalance_data_loader import load_and_combine_imbalance_csvs
        
        csv_content = """Timeinterval Start Loc;Price Shortage;Price Surplus;Regulation State
2024-10-01 00:00:00;50.5;40.5;1
2024-10-01 01:00:00;invalid;41.0;1
2024-10-01 02:00:00;60.0;42.0;1
"""
        csv_file = temp_imbalance_dir / "imbalance.csv"
        csv_file.write_text(csv_content)
        
        result = load_and_combine_imbalance_csvs(temp_imbalance_dir)
        
        # Invalid price should be converted to NaN, then forward-filled
        # The first valid value (50.5) should be carried forward
        assert result["shortage_price"].iloc[1] == 50.5  # forward-filled from row 0
        assert result["surplus_price"].iloc[1] == 41.0  # valid value
        assert result["shortage_price"].iloc[2] == 60.0  # valid value
        
        # Verify no NaN values remain after imputation
        assert result["shortage_price"].isnull().sum() == 0
        assert result["surplus_price"].isnull().sum() == 0


class TestBuildUnifiedImbalanceDataset:
    """Tests for build_unified_imbalance_dataset function."""
    
    def test_build_unified_dataset(self, temp_imbalance_dir, tmp_path, sample_imbalance_csv_content):
        """Test building unified imbalance dataset."""
        from app.services.imbalance_data_loader import build_unified_imbalance_dataset
        
        # Create CSV file in source dir
        csv_file = temp_imbalance_dir / "imbalance.csv"
        csv_file.write_text(sample_imbalance_csv_content)
        
        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result_path = build_unified_imbalance_dataset(
            source_dir=temp_imbalance_dir,
            output_dir=output_dir,
            output_filename="test_unified.csv"
        )
        
        assert result_path.exists()
        assert result_path.name == "test_unified.csv"
        
        # Verify contents
        df = pd.read_csv(result_path)
        assert len(df) == 3
        assert "shortage_price" in df.columns
    
    def test_build_creates_output_directory(self, temp_imbalance_dir, tmp_path, sample_imbalance_csv_content):
        """Test that output directory is created if it doesn't exist."""
        from app.services.imbalance_data_loader import build_unified_imbalance_dataset
        
        csv_file = temp_imbalance_dir / "imbalance.csv"
        csv_file.write_text(sample_imbalance_csv_content)
        
        output_dir = tmp_path / "nonexistent" / "nested"
        
        result_path = build_unified_imbalance_dataset(
            source_dir=temp_imbalance_dir,
            output_dir=output_dir
        )
        
        assert output_dir.exists()
        assert result_path.exists()

