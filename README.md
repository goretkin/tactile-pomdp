tactile-pomdp
=============

Current run instructions

```
$ ipython
In [1]: %gui qt
In [2]: %run place_framework_cleanup.pi
```
%gui qt wasn't necessary before, but since using homebrew with latest PyQT4, Python 2.7.11, IPython 4.0.1, etc, ipython will freeze up if the GUI window is in the background.
It's still sluggish, but with %gui qt it doesn't freeze.  See [here](http://ipython.readthedocs.org/en/stable/config/eventloops.html?highlight=event%20loop) for more info