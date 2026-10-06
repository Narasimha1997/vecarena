package vecarena

import "syscall"

// mapRegion reserves size bytes of zeroed, page-aligned memory outside the Go
// heap. Darwin has no MAP_NORESERVE, but anonymous pages are still
// committed lazily. HugePages is ignored.
func mapRegion(size int, opt Options) ([]byte, error) {
	mem, err := syscall.Mmap(-1, 0, size,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		return nil, err
	}
	if opt.Lock {
		if err := syscall.Mlock(mem); err != nil {
			syscall.Munmap(mem)
			return nil, err
		}
	}
	return mem, nil
}

func unmapRegion(mem []byte) error { return syscall.Munmap(mem) }
