import os
import paramiko
import ftplib
from pathlib import Path
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchFolderMatcher:
    def __init__(self, nas_host, nas_username, nas_password, connection_type='sftp'):
        self.nas_host = nas_host
        self.nas_username = nas_username
        self.nas_password = nas_password
        self.connection_type = connection_type
        
        self.ssh_client = None
        self.sftp_client = None
        self.ftp_client = None
    
    def connect(self):
        """Connect to NAS"""
        try:
            if self.connection_type == 'sftp':
                logger.info(f"Connecting to {self.nas_host} via SFTP...")
                self.ssh_client = paramiko.SSHClient()
                self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                self.ssh_client.connect(
                    hostname=self.nas_host,
                    port=22,
                    username=self.nas_username,
                    password=self.nas_password,
                    timeout=30
                )
                self.sftp_client = self.ssh_client.open_sftp()
                logger.info("✅ SFTP connection successful!")
                
            elif self.connection_type == 'ftp':
                logger.info(f"Connecting to {self.nas_host} via FTP...")
                self.ftp_client = ftplib.FTP()
                self.ftp_client.connect(self.nas_host, 21)
                self.ftp_client.login(self.nas_username, self.nas_password)
                logger.info("✅ FTP connection successful!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def find_matching_folders(self, local_base_path, folder_pattern):
        """
        Find all folders matching the pattern
        
        Args:
            local_base_path: Base path to search (e.g., "citystreet_sunny_day")
            folder_pattern: Pattern to match (e.g., "citystreet_sunny_day_")
        
        Returns:
            List of tuples: (folder_name, timestamp_part)
        """
        try:
            base_path = Path(local_base_path)
            if not base_path.exists():
                logger.error(f"❌ Local base path doesn't exist: {local_base_path}")
                return []
            
            matching_folders = []
            
            # Look for folders matching the pattern
            for item in base_path.iterdir():
                if item.is_dir() and item.name.startswith(folder_pattern):
                    # Extract the timestamp part (XXXXX)
                    timestamp_part = item.name[len(folder_pattern):]
                    matching_folders.append((item.name, timestamp_part))
            
            logger.info(f"🔍 Found {len(matching_folders)} matching folders:")
            for folder_name, timestamp in matching_folders:
                logger.info(f"  📁 {folder_name} -> timestamp: {timestamp}")
            
            return matching_folders
            
        except Exception as e:
            logger.error(f"❌ Failed to find matching folders: {e}")
            return []
    
    def create_remote_directory(self, remote_path):
        """Create directory on NAS recursively"""
        try:
            if self.connection_type == 'sftp':
                try:
                    self.sftp_client.stat(remote_path)
                    return True
                except FileNotFoundError:
                    # Create directory recursively
                    parts = remote_path.strip('/').split('/')
                    current_path = ''
                    for part in parts:
                        if part:  # Skip empty parts
                            current_path += '/' + part
                            try:
                                self.sftp_client.mkdir(current_path)
                                logger.info(f"✅ Created directory: {current_path}")
                            except OSError:
                                pass  # Directory might already exist
            
            elif self.connection_type == 'ftp':
                try:
                    self.ftp_client.cwd(remote_path)
                    self.ftp_client.cwd('/')  # Return to root
                    return True
                except ftplib.error_perm:
                    # Create directory recursively
                    parts = remote_path.strip('/').split('/')
                    current_path = ''
                    for part in parts:
                        if part:  # Skip empty parts
                            current_path += '/' + part
                            try:
                                self.ftp_client.mkd(current_path)
                                logger.info(f"✅ Created directory: {current_path}")
                            except ftplib.error_perm:
                                pass  # Directory might already exist
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create remote directory {remote_path}: {e}")
            return False
    
    def upload_file(self, local_file_path, remote_file_path):
        """Upload a single file"""
        try:
            local_path = Path(local_file_path)
            file_size = local_path.stat().st_size
            
            if self.connection_type == 'sftp':
                self.sftp_client.put(str(local_path), remote_file_path)
            elif self.connection_type == 'ftp':
                with open(local_path, 'rb') as file:
                    self.ftp_client.storbinary(f'STOR {remote_file_path}', file)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Upload failed for {local_file_path}: {e}")
            return False
    
    def transfer_target_folder(self, local_source_path, remote_dest_path, target_subfolder):
        """Transfer the target subfolder from local to remote"""
        try:
            # Find the target subfolder - check multiple possible locations
            possible_paths = [
                Path(local_source_path) / target_subfolder,  # Direct: folder/bev_pcd
                Path(local_source_path) / "colored_pcd_img" / target_subfolder,  # Inside colored_pcd_img: folder/colored_pcd_img/bev_pcd
            ]
            
            target_folder_path = None
            for path in possible_paths:
                if path.exists():
                    target_folder_path = path
                    logger.info(f"    📍 Found target at: {path}")
                    break
            
            if target_folder_path is None:
                logger.error(f"❌ Target folder '{target_subfolder}' not found in any of these locations:")
                for path in possible_paths:
                    logger.error(f"    ❌ {path}")
                return False
            
            # Create remote destination (create the bev_pcd subfolder)
            remote_target_path = f"{remote_dest_path.rstrip('/')}/{target_subfolder}"
            self.create_remote_directory(remote_target_path)
            
            # Transfer all files recursively
            success_count = 0
            total_files = 0
            
            for root, dirs, files in os.walk(target_folder_path):
                # Create corresponding remote directory structure
                relative_path = os.path.relpath(root, target_folder_path)
                
                if relative_path == '.':
                    current_remote_path = remote_target_path
                else:
                    current_remote_path = f"{remote_target_path}/{relative_path}".replace('\\', '/')
                    self.create_remote_directory(current_remote_path)
                
                # Transfer files in current directory
                for file_name in files:
                    total_files += 1
                    local_file = Path(root) / file_name
                    remote_file = f"{current_remote_path}/{file_name}"
                    
                    if self.upload_file(str(local_file), remote_file):
                        success_count += 1
                    
                    # Progress update every 50 files
                    if total_files % 50 == 0:
                        logger.info(f"    📊 Progress: {success_count}/{total_files} files")
            
            logger.info(f"    ✅ Completed: {success_count}/{total_files} files transferred")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Transfer failed for {local_source_path}: {e}")
            return False
    
    def batch_transfer_matching_folders(self, local_base_pattern, remote_base_pattern, target_subfolder):
        """
        Transfer target subfolder from all matching local folders to corresponding NAS folders
        
        Args:
            local_base_pattern: Local pattern (e.g., "citystreet_sunny_day/citystreet_sunny_day_")
            remote_base_pattern: Remote pattern (e.g., "/1323LAB_FTP/Labeled_RealCar_Dataset/2024-06-12/citystreet_sunny_day/citystreet_sunny_day_")
            target_subfolder: Subfolder to transfer (e.g., "bev_pcd")
        """
        try:
            # Parse the local pattern
            local_parts = local_base_pattern.split('/')
            local_base_path = '/'.join(local_parts[:-1])  # "citystreet_sunny_day"
            local_folder_prefix = local_parts[-1]         # "citystreet_sunny_day_"
            
            logger.info(f"🔍 Searching in: {local_base_path}")
            logger.info(f"🔍 Looking for folders starting with: {local_folder_prefix}")
            
            # Find all matching folders
            matching_folders = self.find_matching_folders(local_base_path, local_folder_prefix)
            
            if not matching_folders:
                logger.error("❌ No matching folders found!")
                return False
            
            # Process each matching folder
            total_folders = len(matching_folders)
            successful_transfers = 0
            
            logger.info(f"🚀 Starting batch transfer of {total_folders} folders...")
            
            for i, (folder_name, timestamp) in enumerate(matching_folders, 1):
                logger.info(f"\n📁 [{i}/{total_folders}] Processing: {folder_name}")
                
                # Build paths
                local_source_path = f"{local_base_path}/{folder_name}"
                remote_dest_path = f"{remote_base_pattern}{timestamp}"
                
                logger.info(f"    📤 Local:  {local_source_path}/colored_pcd_img/{target_subfolder}/")
                logger.info(f"    📥 Remote: {remote_dest_path}/{target_subfolder}/")
                
                # Check if local source exists
                if not Path(local_source_path).exists():
                    logger.error(f"    ❌ Local folder doesn't exist: {local_source_path}")
                    continue
                
                # Check if target subfolder exists in any possible location
                possible_target_paths = [
                    Path(local_source_path) / target_subfolder,
                    Path(local_source_path) / "colored_pcd_img" / target_subfolder,
                ]
                
                target_exists = any(path.exists() for path in possible_target_paths)
                if not target_exists:
                    logger.error(f"    ❌ Target subfolder '{target_subfolder}' doesn't exist in:")
                    for path in possible_target_paths:
                        logger.error(f"        {path}")
                    continue
                
                # Transfer the target subfolder
                if self.transfer_target_folder(local_source_path, remote_dest_path, target_subfolder):
                    successful_transfers += 1
                    logger.info(f"    🎉 Success!")
                else:
                    logger.error(f"    ❌ Failed!")
            
            # Final summary
            logger.info(f"\n🏁 BATCH TRANSFER COMPLETED!")
            logger.info(f"📊 Results: {successful_transfers}/{total_folders} folders transferred successfully")
            
            if successful_transfers == total_folders:
                logger.info("🎉 ALL TRANSFERS SUCCESSFUL!")
            elif successful_transfers > 0:
                logger.info(f"⚠️  Partial success: {total_folders - successful_transfers} folders failed")
            else:
                logger.error("❌ ALL TRANSFERS FAILED!")
            
            return successful_transfers > 0
            
        except Exception as e:
            logger.error(f"❌ Batch transfer failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from NAS"""
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()
        if self.ftp_client:
            try:
                self.ftp_client.quit()
            except:
                self.ftp_client.close()
        logger.info("🔌 Disconnected from NAS")

def run_batch_transfer():
    """Run batch transfer for matching folders"""
    
    print("=" * 80)
    print("🔄 BATCH FOLDER MATCHER AND TRANSFER")
    print("=" * 80)
    
    # ===========================================
    # CONFIGURE THESE SETTINGS
    # ===========================================
    
    # NAS Settings
    NAS_HOST = "140.124.182.50"  # or your NAS IP like "192.168.1.100"
    NAS_USERNAME = "113c52012"              # Your NAS username
    NAS_PASSWORD = "Mama7106!"              # Your NAS password
    CONNECTION_TYPE = "sftp"                    # or "ftp"
    
    # Pattern Settings
    LOCAL_BASE_PATTERN = "citystreet_sunny_day/citystreet_sunny_day_"
    REMOTE_BASE_PATTERN = "/1323LAB_FTP/Labeled_RealCar_Dataset/2024-06-12/citystreet_sunny_day/citystreet_sunny_day_"
    TARGET_SUBFOLDER = "bev_pcd"  # or "colored_pcd_img" or whatever you want to transfer
    
    # ===========================================
    
    print(f"🔍 Local pattern:    {LOCAL_BASE_PATTERN}XXXXX")
    print(f"🎯 Remote pattern:   {REMOTE_BASE_PATTERN}XXXXX")
    print(f"📁 Target subfolder: {TARGET_SUBFOLDER}")
    print(f"🔗 Connection:       {CONNECTION_TYPE.upper()}")
    print()
    
    # Create transfer instance
    transfer = BatchFolderMatcher(NAS_HOST, NAS_USERNAME, NAS_PASSWORD, CONNECTION_TYPE)
    
    # Run the batch transfer
    if transfer.connect():
        print()
        success = transfer.batch_transfer_matching_folders(
            LOCAL_BASE_PATTERN,
            REMOTE_BASE_PATTERN, 
            TARGET_SUBFOLDER
        )
        
        if success:
            print("\n🎉 BATCH TRANSFER COMPLETED SUCCESSFULLY!")
        else:
            print("\n❌ BATCH TRANSFER FAILED!")
        
        transfer.disconnect()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    run_batch_transfer()

# Example of what this script does:
# 
# 1. Finds local folders like:
#    - citystreet_sunny_day/citystreet_sunny_day_2024-06-12-08-59-06/colored_pcd_img/bev_pcd/
#    - citystreet_sunny_day/citystreet_sunny_day_2024-06-12-09-00-07/colored_pcd_img/bev_pcd/
#    - citystreet_sunny_day/citystreet_sunny_day_2024-06-12-09-01-19/colored_pcd_img/bev_pcd/
#
# 2. For each folder with timestamp XXXXX, transfers:
#    FROM: citystreet_sunny_day/citystreet_sunny_day_XXXXX/colored_pcd_img/bev_pcd/
#    TO:   /1323LAB_FTP/Labeled_RealCar_Dataset/2024-06-12/citystreet_sunny_day/citystreet_sunny_day_XXXXX/bev_pcd/
#
# 3. Creates the full directory structure on NAS automatically including the bev_pcd subfolder