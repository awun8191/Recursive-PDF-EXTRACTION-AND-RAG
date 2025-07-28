from config import load_config

config = load_config()

file_path = config.sample_pdf_path.split("/")
print(file_path)
