#include <fcntl.h>
#include <linux/memfd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/vfs.h>
#include <unistd.h>

#include "zdtmtst.h"

const char *test_doc = "many memfd mappings";
const char *test_author = "Dan Feigin <dfeigin@nvidia.com>";

#define MEMFD_COUNT 16
#define LEN	    PAGE_SIZE

#define err(exitcode, msg, ...)                \
	({                                     \
		pr_perror(msg, ##__VA_ARGS__); \
		exit(exitcode);                \
	})

struct memfd_mapping {
	int fd;
	void *shared;
	void *private;
	int fl_flags;
	int fd_flags;
	off_t pos;
	dev_t shared_dev;
	dev_t private_dev;
};

static struct memfd_mapping mappings[MEMFD_COUNT];

static int _memfd_create(const char *name, unsigned int flags)
{
	return syscall(SYS_memfd_create, name, flags);
}

static void fill_pattern(char *buf, int idx, int variant)
{
	unsigned int seed = 0x9e3779b9U ^ (idx * 0x10001U) ^ (variant * 0x45d9f3bU);
	size_t i;

	for (i = 0; i < LEN; i++) {
		seed = seed * 1103515245U + 12345U;
		buf[i] = (char)(seed >> 24);
	}
}

static int check_pattern(const void *addr, int idx, int variant, const char *what)
{
	char expected[LEN];

	fill_pattern(expected, idx, variant);
	if (memcmp(addr, expected, LEN)) {
		fail("%s content mismatch for memfd %d", what, idx);
		return -1;
	}

	return 0;
}

static int check_fd_pattern(int fd, int idx, int variant)
{
	char buf[LEN];

	if (pread(fd, buf, sizeof(buf), 0) != sizeof(buf)) {
		fail("read problem for memfd %d", idx);
		return -1;
	}

	return check_pattern(buf, idx, variant, "fd");
}

int main(int argc, char *argv[])
{
	char buf[LEN];
	int i;

	test_init(argc, argv);

	for (i = 0; i < MEMFD_COUNT; i++) {
		struct memfd_mapping *m = &mappings[i];
		char name[32];

		snprintf(name, sizeof(name), "many-memfd-%d", i);
		m->fd = _memfd_create(name, MFD_CLOEXEC);
		if (m->fd < 0)
			err(1, "Can't call memfd_create");

		fill_pattern(buf, i, 0);
		if (write(m->fd, buf, sizeof(buf)) != sizeof(buf))
			err(1, "write error");

		if (fcntl(m->fd, F_SETFL, O_APPEND) < 0)
			err(1, "Can't set fl flags");

		m->fl_flags = fcntl(m->fd, F_GETFL);
		if (m->fl_flags == -1)
			err(1, "Can't get fl flags");

		m->fd_flags = fcntl(m->fd, F_GETFD);
		if (m->fd_flags == -1)
			err(1, "Can't get fd flags");

		m->pos = i * 17;
		if (lseek(m->fd, m->pos, SEEK_SET) < 0)
			err(1, "seek error");

		m->shared = mmap(NULL, LEN, PROT_READ | PROT_WRITE, MAP_SHARED, m->fd, 0);
		if (m->shared == MAP_FAILED)
			err(1, "Can't mmap shared");

		m->private = mmap(NULL, LEN, PROT_READ | PROT_WRITE, MAP_PRIVATE, m->fd, 0);
		if (m->private == MAP_FAILED)
			err(1, "Can't mmap private");

		m->shared_dev = get_mapping_dev(m->shared);
		if (m->shared_dev == (dev_t)-1) {
			fail("Can't get shared mapping dev");
			return 1;
		}

		m->private_dev = get_mapping_dev(m->private);
		if (m->private_dev == (dev_t)-1) {
			fail("Can't get private mapping dev");
			return 1;
		}

		if (check_pattern(m->shared, i, 0, "shared"))
			return 1;
		if (check_pattern(m->private, i, 0, "private"))
			return 1;

		fill_pattern(m->private, i, 1);
	}

	test_daemon();
	test_waitsig();

	for (i = 0; i < MEMFD_COUNT; i++) {
		struct memfd_mapping *m = &mappings[i];
		int fl_flags, fd_flags;

		fl_flags = fcntl(m->fd, F_GETFL);
		if (fl_flags == -1)
			err(1, "Can't get fl flags");
		if (m->fl_flags != fl_flags) {
			fail("fl flags differ for memfd %d", i);
			return 1;
		}

		fd_flags = fcntl(m->fd, F_GETFD);
		if (fd_flags == -1)
			err(1, "Can't get fd flags");
		if (m->fd_flags != fd_flags) {
			fail("fd flags differ for memfd %d", i);
			return 1;
		}

		if (m->pos != lseek(m->fd, 0, SEEK_CUR)) {
			fail("position differs for memfd %d", i);
			return 1;
		}

		if (m->shared_dev != get_mapping_dev(m->shared)) {
			fail("shared mapping dev mismatch for memfd %d", i);
			return 1;
		}

		if (m->private_dev != get_mapping_dev(m->private)) {
			fail("private mapping dev mismatch for memfd %d", i);
			return 1;
		}

		if (check_fd_pattern(m->fd, i, 0))
			return 1;
		if (check_pattern(m->shared, i, 0, "shared"))
			return 1;
		if (check_pattern(m->private, i, 1, "private"))
			return 1;

		fill_pattern(m->shared, i, 2);
		if (check_fd_pattern(m->fd, i, 2))
			return 1;
		if (check_pattern(m->private, i, 1, "private"))
			return 1;

		fill_pattern(m->private, i, 3);
		if (check_fd_pattern(m->fd, i, 2))
			return 1;
		if (check_pattern(m->shared, i, 2, "shared"))
			return 1;
	}

	pass();

	return 0;
}
