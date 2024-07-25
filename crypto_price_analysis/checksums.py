from pathlib import Path
import hashlib


def verify_file_checksum(file_path: Path) -> bool:
	try:
		with open(file_path, 'rb') as file:
			file_hash = hashlib.file_digest(file, 'sha256').hexdigest()

		with open(file_path.parent / (file_path.name + '.CHECKSUM'), 'r') as checksum_file:
			provided_hash = checksum_file.read(64)

	except FileNotFoundError:
		return False

	return file_hash == provided_hash


def create_file_checksum(file_path: Path) -> None:

	with open(file_path, 'rb') as file:
		file_hash = hashlib.file_digest(file, 'sha256').hexdigest()

	with open(file_path.parent / (file_path.name + '.CHECKSUM'), 'w') as checksum_file:
		checksum_file.write(file_hash + '  ' + file_path.name)