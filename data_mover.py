import os
import sys

def create_directories(filename):
    # Extract filename without extension
    file_basename = os.path.splitext(filename)[0]
    
    # Create directory with the same name as file
    os.makedirs(file_basename)
    
    # Create "raw" and "processed" subdirectories
    os.makedirs(os.path.join(file_basename, 'raw'))
    os.makedirs(os.path.join(file_basename, 'processed'))

def move_file(filename):
    # Extract filename without extension
    file_basename = os.path.splitext(filename)[0]
    
    # Move file to "raw" subdirectory with name "data.pt"
    os.rename(filename, os.path.join(file_basename, 'raw', 'data.pt'))

if __name__ == '__main__':
    # Check if filename argument is provided
    if len(sys.argv) < 2:
        print("Please provide a filename as an argument")
        sys.exit(1)

    # Extract filename from argument
    filename = sys.argv[1]

    # Create directories and move file
    create_directories(filename)
    move_file(filename)
