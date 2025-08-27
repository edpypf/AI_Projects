import os
import sys
import time
import shutil
import tempfile
import psutil
import gc
from pathlib import Path
from typing import List, Optional

class FileCleanupUtility:
    """Utility class to handle file cleanup and process management for RAG applications."""
    
    def __init__(self):
        self.temp_directories = []
        self.locked_files = []
    
    def find_python_processes(self) -> List[dict]:
        """Find all running Python processes."""
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': proc.info['cmdline']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return python_processes
    
    def kill_process_by_pid(self, pid: int, force: bool = False) -> bool:
        """Kill a process by PID."""
        try:
            process = psutil.Process(pid)
            if force:
                process.kill()
            else:
                process.terminate()
            
            # Wait for process to terminate
            process.wait(timeout=10)
            print(f"Successfully terminated process {pid}")
            return True
        except psutil.NoSuchProcess:
            print(f"Process {pid} not found")
            return True
        except psutil.TimeoutExpired:
            if not force:
                print(f"Process {pid} didn't terminate gracefully, forcing...")
                return self.kill_process_by_pid(pid, force=True)
            else:
                print(f"Failed to force kill process {pid}")
                return False
        except Exception as e:
            print(f"Error killing process {pid}: {e}")
            return False
    
    def find_temp_directories(self, pattern: str = "tmp") -> List[str]:
        """Find temporary directories that might contain locked files."""
        temp_dirs = []
        temp_root = tempfile.gettempdir()
        
        try:
            for item in os.listdir(temp_root):
                item_path = os.path.join(temp_root, item)
                if os.path.isdir(item_path) and pattern in item:
                    temp_dirs.append(item_path)
        except PermissionError:
            print(f"Permission denied accessing {temp_root}")
        
        return temp_dirs
    
    def force_remove_directory(self, directory: str, max_attempts: int = 5) -> bool:
        """Force remove a directory with multiple attempts."""
        if not os.path.exists(directory):
            return True
        
        for attempt in range(max_attempts):
            try:
                # Force garbage collection
                gc.collect()
                
                # Wait a bit for file handles to close
                time.sleep(0.5)
                
                # Try to remove the directory
                shutil.rmtree(directory, ignore_errors=False)
                print(f"Successfully removed directory: {directory}")
                return True
                
            except PermissionError as e:
                print(f"Attempt {attempt + 1}: Permission error removing {directory}: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(1)
                    continue
                else:
                    # Last attempt - try to handle individual files
                    return self._force_remove_files_individually(directory)
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error removing {directory}: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(1)
                    continue
        
        return False
    
    def _force_remove_files_individually(self, directory: str) -> bool:
        """Try to remove files individually when directory removal fails."""
        try:
            for root, dirs, files in os.walk(directory, topdown=False):
                # Remove files first
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.chmod(file_path, 0o777)  # Change permissions
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Failed to remove file {file_path}: {e}")
                
                # Remove directories
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        os.rmdir(dir_path)
                    except Exception as e:
                        print(f"Failed to remove directory {dir_path}: {e}")
            
            # Finally remove the root directory
            os.rmdir(directory)
            return True
            
        except Exception as e:
            print(f"Failed to remove directory individually: {e}")
            return False
    
    def cleanup_chroma_temp_files(self) -> bool:
        """Clean up Chroma-related temporary files."""
        temp_dirs = self.find_temp_directories()
        success = True
        
        for temp_dir in temp_dirs:
            try:
                # Look for Chroma-related files
                for root, dirs, files in os.walk(temp_dir):
                    chroma_files = [f for f in files if any(ext in f.lower() for ext in ['.bin', '.pkl', 'chroma', 'faiss'])]
                    if chroma_files:
                        print(f"Found Chroma-related files in {root}: {chroma_files}")
                        if not self.force_remove_directory(temp_dir):
                            success = False
                        break
            except Exception as e:
                print(f"Error checking directory {temp_dir}: {e}")
                success = False
        
        return success
    
    def emergency_cleanup(self) -> bool:
        """Emergency cleanup - kill Python processes and clean temp files."""
        print("Starting emergency cleanup...")
        
        # Step 1: Find and optionally kill Python processes
        python_processes = self.find_python_processes()
        print(f"Found {len(python_processes)} Python processes:")
        
        for proc in python_processes:
            print(f"  PID: {proc['pid']}, Name: {proc['name']}")
            if proc['cmdline']:
                cmdline_str = ' '.join(proc['cmdline'][:3])  # Show first 3 args
                print(f"    Command: {cmdline_str}")
        
        # Step 2: Clean up temporary files
        print("\nCleaning up temporary files...")
        cleanup_success = self.cleanup_chroma_temp_files()
        
        # Step 3: Force garbage collection
        gc.collect()
        
        return cleanup_success

def main():
    """Main function to run the cleanup utility."""
    cleanup = FileCleanupUtility()
    
    print("File Cleanup Utility")
    print("===================")
    
    # Show current Python processes
    processes = cleanup.find_python_processes()
    if processes:
        print(f"\nFound {len(processes)} Python processes:")
        for i, proc in enumerate(processes):
            print(f"{i+1}. PID: {proc['pid']}, Name: {proc['name']}")
    
    # Ask user what to do
    print("\nOptions:")
    print("1. Emergency cleanup (clean temp files)")
    print("2. Kill specific Python process")
    print("3. Kill all Python processes and cleanup")
    print("4. Exit")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            cleanup.emergency_cleanup()
        elif choice == "2":
            if not processes:
                print("No Python processes found.")
                return
            
            try:
                proc_num = int(input(f"Enter process number (1-{len(processes)}): ")) - 1
                if 0 <= proc_num < len(processes):
                    pid = processes[proc_num]['pid']
                    cleanup.kill_process_by_pid(pid)
                else:
                    print("Invalid process number.")
            except ValueError:
                print("Invalid input.")
        elif choice == "3":
            for proc in processes:
                cleanup.kill_process_by_pid(proc['pid'])
            cleanup.emergency_cleanup()
        elif choice == "4":
            print("Exiting...")
        else:
            print("Invalid choice.")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
