import unittest
from utils import metrics, data_loading
# Import other necessary modules and functions


class TestMetrics(unittest.TestCase):
    def test_psnr(self):
        import torch
        # Create dummy data for prediction and target images
        pred = torch.full((1, 3, 10, 10), 0.5)
        target = torch.full((1, 3, 10, 10), 0.5)
        # Expected PSNR value for identical images should be very high (infinity in theory, but capped by max_pixel and eps)
        expected_psnr = metrics.psnr(pred, target)
        self.assertTrue(expected_psnr > 50)  # Arbitrary high value to indicate close match

class TestDataLoading(unittest.TestCase):
    def test_dataset_length(self):
        dataset = data_loading.PairedImageDataset(directory='path/to/test/dataset', before_transform=None, after_transform=None)
        # Assuming you know the exact number of image pairs in your test dataset
        expected_length = 100  # Example value
        self.assertEqual(len(dataset), expected_length)
if __name__ == '__main__':
    unittest.main()
