import torch, math, random, matplotlib.pyplot as plt
torch.manual_seed(0)

device     = 'cpu'
STEPS      = 50_000
EVAL_EVERY = 200
MIN_HIDDEN = 0
MAX_HIDDEN = 3
MUT_SIGMA  = 0.01
WIDTH      = 16

# ---------- data ----------
x = torch.linspace(0, 1, 128).unsqueeze(1)
y = torch.sin(2 * math.pi * x)

# ---------- network ----------
# each layer is (W, b) with shapes:
# head: (1, WIDTH), (WIDTH)
# tail: (WIDTH, 1), (1)
# hidden: (WIDTH, WIDTH), (WIDTH)

head   = (torch.randn(1, WIDTH), torch.randn(WIDTH))
tail   = (torch.randn(WIDTH, 1), torch.randn(1))
hidden = []  # list of (W, b) for hidden layers

def forward(x):
    W, b = head
    h = torch.tanh(x @ W + b)
    for W, b in hidden:
        h = torch.tanh(h @ W + b)
    W, b = tail
    return h @ W + b

def loss():
    return ((forward(x) - y) ** 2).mean()

def rand_wb():
    W = torch.randn(WIDTH, WIDTH)
    b = torch.randn(WIDTH)
    return (W, b)

def mutate_wb(wb, sigma):
    W, b = wb
    return (W + torch.randn_like(W) * sigma,
            b + torch.randn_like(b) * sigma)

# ---------- helpers ----------
def choose_action():
    actions = ['mutate']
    if len(hidden) < MAX_HIDDEN:
        actions.append('add')
    if len(hidden) > MIN_HIDDEN:
        actions.append('remove')
    return random.choice(actions)

# ---------- initial ----------
current_mse = loss()
print(f"step {0:5d}  hidden {len(hidden)}   MSE {current_mse:.6f}")

# ---------- training ----------
for step in range(1, STEPS + 1):
    # store old state
    old_head   = (head[0].clone(), head[1].clone())
    old_tail   = (tail[0].clone(), tail[1].clone())
    old_hidden = [(W.clone(), b.clone()) for (W, b) in hidden]

    action = choose_action()

    if action == 'add':
        pos = random.randint(0, len(hidden))
        hidden.insert(pos, rand_wb())
    elif action == 'remove':
        hidden.pop(random.randint(0, len(hidden) - 1))
    elif action == 'mutate':
        zone = random.choice(['head', 'tail', 'hidden'] if hidden else ['head', 'tail'])
        if zone == 'head':
            head = mutate_wb(head, MUT_SIGMA)
        elif zone == 'tail':
            tail = mutate_wb(tail, MUT_SIGMA)
        else:
            pos = random.randint(0, len(hidden) - 1)
            hidden[pos] = mutate_wb(hidden[pos], MUT_SIGMA)

    new_mse = loss()
    if new_mse < current_mse:
        current_mse = new_mse
    else:
        head, tail, hidden = old_head, old_tail, old_hidden

    if step % EVAL_EVERY == 0:
        print(f"step {step:5d}  hidden {len(hidden)}   MSE {current_mse:.6f}")

print("\nFinal hidden layers:", len(hidden), "Final MSE:", current_mse.item())

# ---------- plot ----------
with torch.no_grad():
    x_plot = torch.linspace(0, 1, 300).unsqueeze(1)
    y_hat = forward(x_plot).squeeze()
plt.plot(torch.linspace(0, 1, 300), torch.sin(2 * math.pi * torch.linspace(0, 1, 300)), label='target')
plt.plot(torch.linspace(0, 1, 300), y_hat, label='tapeworm')
plt.legend(); plt.show()
