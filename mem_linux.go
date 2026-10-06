package vecarena

import "syscall"

// mapRegion reserves size bytes of zeroed, page-aligned memory outside the Go
// heap. MAP_NORESERVE commits RAM lazily, page by page, on first write.
func mapRegion(size int, opt Options) ([]byte, error) {
	mem, err := syscall.Mmap(-1, 0, size,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANON|syscall.MAP_NORESERVE)
	if err != nil {
		return nil, err
	}
	if opt.HugePages {
		_ = syscall.Madvise(mem, syscall.MADV_HUGEPAGE) // best effort
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
