#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/utsname.h>

#include <stdio.h>
#include <string.h>

static int (*real_gethostname)(char *name, size_t len);

int uname(struct utsname *buf)
{
 int ret;

 ret = syscall(SYS_uname, buf);

 strcpy(buf->nodename, "localhost");

 return ret;
}

int gethostname(char *name, size_t len)
{
  const char *val;

  /* Override hostname */
  val = "localhost";
  if (val != NULL)
  {
    strncpy(name, val, len);
    return 0;
  }

  /* Call real gethostname() */
  return real_gethostname(name, len);
}