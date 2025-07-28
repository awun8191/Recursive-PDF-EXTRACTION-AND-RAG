import boto3
import os
from botocore.client import Config
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

class CloudflareR2:
    def __init__(self):
        self.account_id = "c1719c3cf4696ae260e6a5f57b1f3100"
        self.access_key_id = "6d6c54828aced5792f4a1ea6344691fd"
        self.secret_access_key = "39a04416b64bf8df8d0e009346836ba4ff4382c1d423b35228a5d8717ff26ec4"
        self.bucket_name = "study-materials"
        
        if not all([self.access_key_id, self.secret_access_key, self.bucket_name]):
            raise ValueError("Cloudflare R2 credentials are not set.")

        self.s3_client = boto3.client(
            's3',
            endpoint_url=f'https://{self.account_id}.r2.cloudflarestorage.com',  # Updated endpoint
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(signature_version='s3v4')
        )

    def upload_file(self, file_path: str, file_name: str) -> str:
        """Uploads a file to Cloudflare R2 and returns its public URL."""
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, file_name)
            # Updated to use the free Cloudflare domain pattern
            public_url = f"https://{self.bucket_name}.{self.account_id}.r2.dev/{file_name}"
            return public_url
        except (NoCredentialsError, PartialCredentialsError):
            print("Error: AWS credentials not found.")
            raise
        except Exception as e:
            print(f"Error uploading file to R2: {e}")
            raise

if __name__ == "__main__":
    # Example usage (requires environment variables to be set)
    # You can test this by creating a dummy file and setting up your .env file
    if os.getenv("CLOUDFLARE_ACCOUNT_ID"):
        r2_service = CloudflareR2()
        # Create a dummy file for testing
        with open("test.txt", "w") as f:
            f.write("This is a test file.")
        
        try:
            url = r2_service.upload_file("test.txt", "test.txt")
            print(f"File uploaded successfully. URL: {url}")
        except Exception as e:
            print(f"Upload failed: {e}")
        finally:
            os.remove("test.txt")