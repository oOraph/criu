/*
 * proc-pause - pause and resume a process tree via SIGSTOP/SIGCONT
 *
 * Usage:
 *   proc-pause --pid PID --action pause
 *   proc-pause --pid PID --action resume
 *
 * pause: sends SIGSTOP to the process tree in BFS order (parent before
 *        children) so the parent cannot react to a stopped child before it
 *        is itself stopped.
 *
 * resume: sends SIGCONT in reverse BFS order (children before parent) so
 *         children are runnable before the parent can query their state.
 *
 * Intended use around criu dump:
 *   cuda-offload --pid PID --dir DIR --action checkpoint
 *   proc-pause   --pid PID --action pause
 *   criu dump    -t PID -D DIR ...
 *   proc-pause   --pid PID --action resume   # optional; criu restore handles it
 *
 * Timer note: wall-clock timers (alarm, ITIMER_REAL, CLOCK_REALTIME) keep
 * ticking while the process tree is stopped.  Keep the freeze window shorter
 * than the application's watchdog timeout, or raise the timeout before
 * pausing.
 */

#include <dirent.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define pr_info(fmt, ...)   fprintf(stderr, "proc-pause: " fmt, ##__VA_ARGS__)
#define pr_err(fmt, ...)    fprintf(stderr, "proc-pause: ERROR: " fmt, ##__VA_ARGS__)
#define pr_perror(fmt, ...) fprintf(stderr, "proc-pause: ERROR: " fmt ": %s\n", \
				    ##__VA_ARGS__, strerror(errno))

static int append_pid(int **pids, int *n, int *cap, int pid)
{
	if (*n >= *cap) {
		int new_cap = *cap ? *cap * 2 : 16;
		int *tmp = realloc(*pids, (size_t)new_cap * sizeof(int));

		if (!tmp)
			return -1;
		*pids = tmp;
		*cap = new_cap;
	}
	(*pids)[(*n)++] = pid;
	return 0;
}

/*
 * Collect all PIDs in the process subtree rooted at root_pid (BFS).
 * The root is always first; caller must free(*out_pids).
 */
static int collect_pids(int root_pid, int **out_pids, int *out_n)
{
	int *pids = NULL, n = 0, cap = 0;
	int i;

	if (append_pid(&pids, &n, &cap, root_pid) < 0)
		return -1;

	for (i = 0; i < n; i++) {
		char task_path[64];
		DIR *task_dir;
		struct dirent *tid_ent;

		snprintf(task_path, sizeof(task_path), "/proc/%d/task", pids[i]);
		task_dir = opendir(task_path);
		if (!task_dir)
			continue;

		while ((tid_ent = readdir(task_dir)) != NULL) {
			char children_path[320];
			FILE *f;
			int child_pid;

			if (tid_ent->d_name[0] == '.')
				continue;

			snprintf(children_path, sizeof(children_path),
				 "/proc/%d/task/%s/children",
				 pids[i], tid_ent->d_name);
			f = fopen(children_path, "r");
			if (!f)
				continue;

			while (fscanf(f, "%d", &child_pid) == 1) {
				if (append_pid(&pids, &n, &cap, child_pid) < 0) {
					fclose(f);
					closedir(task_dir);
					free(pids);
					return -1;
				}
			}
			fclose(f);
		}
		closedir(task_dir);
	}

	*out_pids = pids;
	*out_n = n;
	return 0;
}

static void usage(const char *prog)
{
	fprintf(stderr,
		"Usage: %s --pid PID --action pause|resume\n"
		"\n"
		"  pause   SIGSTOP the process tree rooted at PID (BFS order,\n"
		"          parent before children).\n"
		"  resume  SIGCONT the process tree rooted at PID (reverse BFS,\n"
		"          children before parent).\n",
		prog);
}

int main(int argc, char **argv)
{
	int pid = 0, i;
	const char *action = NULL;
	int *pids = NULL, n_pids = 0;
	int ret = 0;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--pid") == 0 && i + 1 < argc)
			pid = atoi(argv[++i]);
		else if (strcmp(argv[i], "--action") == 0 && i + 1 < argc)
			action = argv[++i];
		else {
			fprintf(stderr, "Unknown argument: %s\n", argv[i]);
			usage(argv[0]);
			return 1;
		}
	}

	if (!pid || !action) {
		usage(argv[0]);
		return 1;
	}

	if (strcmp(action, "pause") != 0 && strcmp(action, "resume") != 0) {
		fprintf(stderr, "Unknown action: %s\n", action);
		usage(argv[0]);
		return 1;
	}

	if (collect_pids(pid, &pids, &n_pids) < 0) {
		pr_perror("Failed to collect process tree for pid %d", pid);
		return 1;
	}

	pr_info("Found %d process(es) in subtree of pid %d\n", n_pids, pid);

	if (strcmp(action, "pause") == 0) {
		/* BFS order: parent before children */
		for (i = 0; i < n_pids; i++) {
			if (kill(pids[i], SIGSTOP) < 0)
				pr_perror("SIGSTOP pid %d", pids[i]);
			else
				pr_info("pid %d: stopped\n", pids[i]);
		}
	} else {
		/* Reverse BFS: children before parent */
		for (i = n_pids - 1; i >= 0; i--) {
			if (kill(pids[i], SIGCONT) < 0)
				pr_perror("SIGCONT pid %d", pids[i]);
			else
				pr_info("pid %d: resumed\n", pids[i]);
		}
	}

	free(pids);
	return ret;
}
