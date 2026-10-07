"""Download the selected public Kaggle dataset without embedding credentials."""
import kagglehub

if __name__ == '__main__':
    path = kagglehub.dataset_download('chitholian/annotated-potholes-dataset')
    print(f'\nDownloaded dataset folder:\n{path}\n')
    print(f'Next: python prepare_data.py --source "{path}" --output dataset')
