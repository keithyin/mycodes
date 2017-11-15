See http://rll.berkeley.edu/deeprlcourse/docs/hw3.pdf for instructions

The starter code was based on an implementation of Q-learning for Atari
generously provided by Szymon Sidor from OpenAI

# DQN pytorch implentation

pytorch implementation of [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) .

## Model Detail

**model**

```python
conv(in_channels=4, out_channels=32, kernel_size=8, stride=4)
conv(in_channels=32, out_channels=64, kernel_size=4, stride=2)
conv(in_channels=64, out_channels=64, kernel_size=3, stride=1)
linear(in_features=10 * 10 * 64, out_features=512)
logits_layer(in_features=512, out_features=num_actions)
```

* optimizer: RMSprop

**preprocessing**

> aim: reduce the input dimensionality, deal with some artefacts of the Atari 2600 games.

frame preprocessing

* origin frame : (210, 180)  RGB image
* to encode a single frame we take the **maximum value** for each pixel colour value over the **frame** being encoded and the **previous frame**. Aim to remove flickering that is present in games where some objects appear only in even frames while other objects appear only in odd frames.
* extract the **Y channel**, (luminance)
* rescale it to (84*84)



**frame-skipping**

the agent **sees and selects actions** on every **k** frame instead every frame, and its last action repeated on skipped frames.

we use `k=4` for all games.

i.e : choose an action every **k** frame



**behavior policy**

* $\varepsilon-greedy$  $\varepsilon$ annealed linearly from 1.0 to 0.1 over first million frames and fixed at 0.1 thereafter

**target policy**

* greedy



**clip the error term**

> improve the stability of the algorithm

clip $r+\gamma\max\limits_{a'}Q(s',a';\theta^-_i)-Q(s,a;\theta_i)$ to be between -1,1


## Result
Due to the lack of memory resources, i just use the size of Replay buffer 10000.
After episode 850, the agent can beat computer easily.