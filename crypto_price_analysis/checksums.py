from pathlib import Path
import hashlib



def sha256sum(filename):
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()



def verify_file_checksum(file_path: Path) -> bool:
	try:		
		file_hash = sha256sum(file_path)

		with open(file_path.parent / (file_path.name + '.CHECKSUM'), 'r') as checksum_file:
			provided_hash = checksum_file.read(64)

	except FileNotFoundError:
		return False

	return file_hash == provided_hash



def create_file_checksum(file_path: Path) -> None:

	file_hash = sha256sum(file_path)

	with open(file_path.parent / (file_path.name + '.CHECKSUM'), 'w') as checksum_file:
		checksum_file.write(file_hash + '  ' + file_path.name)