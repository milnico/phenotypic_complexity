from mass_spring_robot_config import Morph, Morph_2, Morph_evo, Morph_evolve
import random
import sys
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import taichi as ti
import math
import numpy as np
import os
from copy import copy
import argparse
import os
import pickle

random.seed(0)
np.random.seed(0)
real = ti.f32
ti.init(default_fp=real)


def reset_all():

    global max_steps
    global vis_interval
    global output_vis_interval
    global steps

    global scalar
    global vec

    global loss

    global x
    global v
    global v_inc

    global head_id
    global goal

    global n_objects
    # target_ball = 0
    global elasticity
    global ground_height
    global gravity
    global friction

    global gradient_clip
    global spring_omega
    global damping

    global n_springs
    global spring_anchor_a
    global spring_anchor_b
    global spring_length
    global spring_stiffness
    global spring_actuation
    global spring_phase

    global n_sin_waves
    global weights1
    global bias1

    global n_hidden
    global weights2
    global bias2
    global hidden

    global center
    global workdir
    global act

    max_steps = 4096
    vis_interval = 1
    output_vis_interval = 1
    steps = 1000#2048 // 2#8
    assert steps * 2 <= max_steps

    scalar = lambda: ti.field(dtype=ti.f32)
    vec = lambda: ti.Vector.field(2, dtype=ti.f32)

    loss = scalar()

    x = vec()
    v = vec()
    v_inc = vec()

    head_id = 0
    goal = vec()

    n_objects = 0
    # target_ball = 0
    elasticity = 0.0
    ground_height = 0.1
    gravity = -9.81#-4.8
    friction = 2.5

    gradient_clip = 1
    spring_omega = 10
    damping = 15

    n_springs = 0
    spring_anchor_a = ti.field(ti.i32)
    spring_anchor_b = ti.field(ti.i32)
    spring_length = scalar()
    spring_stiffness = scalar()
    spring_actuation = scalar()
    spring_phase = scalar()

    #print(spring_actuation)
    #print(spring_phase)

    n_sin_waves = 10
    weights1 = scalar()
    bias1 = scalar()

    n_hidden = 32
    weights2 = scalar()
    bias2 = scalar()
    hidden = scalar()

    center = vec()

    act = scalar()
    workdir = os.getcwd()

def n_input_states():
    return n_sin_waves + 4 * n_objects + 2


@ti.layout
def place():
    ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v, v_inc)
    ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                         spring_length, spring_stiffness,
                                         spring_actuation,spring_phase)
    ti.root.dense(ti.ij, (n_hidden, n_input_states())).place(weights1)
    ti.root.dense(ti.i, n_hidden).place(bias1)
    ti.root.dense(ti.ij, (n_springs, n_hidden)).place(weights2)
    ti.root.dense(ti.i, n_springs).place(bias2)
    ti.root.dense(ti.ij, (max_steps, n_hidden)).place(hidden)
    ti.root.dense(ti.ij, (max_steps, n_springs)).place(act)
    ti.root.dense(ti.i, max_steps).place(center)
    ti.root.place(loss, goal)
    ti.root.lazy_grad()


dt = 0.004
learning_rate = 25


@ti.kernel
def compute_center(t: ti.i32,n_objects: ti.i32):
    for _ in range(1):
        c = ti.Vector([0.0, 0.0])
        for i in range(n_objects):
            c += x[t, i]
        center[t] = (1.0 / n_objects) * c



@ti.kernel
def nn1(t: ti.i32):
    for i in range(n_hidden):
        actuation = 0.0
        for j in ti.static(range(n_sin_waves)):
            actuation += weights1[i, j] * ti.sin(spring_omega * t * dt +
                                                 2 * math.pi / n_sin_waves * j)
        for j in ti.static(range(n_objects)):
            offset = x[t, j] - center[t]
            # use a smaller weight since there are too many of them
            actuation += weights1[i, j * 4 + n_sin_waves] * offset[0] * 0.05
            actuation += weights1[i,
                                  j * 4 + n_sin_waves + 1] * offset[1] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 2] * v[t,
                                                                  j][0] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 3] * v[t,
                                                                  j][1] * 0.05
        actuation += weights1[i, n_objects * 4 +
                              n_sin_waves] * (goal[None][0] - center[t][0])
        actuation += weights1[i, n_objects * 4 + n_sin_waves +
                              1] * (goal[None][1] - center[t][1])
        actuation += bias1[i]
        actuation = ti.tanh(actuation)
        hidden[t, i] = actuation


@ti.kernel
def nn2(t: ti.i32):
    for i in range(n_springs):
        actuation = 0.0
        for j in ti.static(range(n_hidden)):
            actuation += weights2[i, j] * hidden[t, j]
        actuation += bias2[i]
        actuation = ti.tanh(actuation)
        #if actuation>0.5:
        #    actuation=0
        #else:
        #    actuation=-1
        act[t, i] = actuation
@ti.kernel
def sin_wave(t: ti.i32,n_springs: ti.i32):
    for i in range(n_springs):

        act[t, i] = ti.sin(t/5+spring_phase[i])
        #print(act[t,i])
        #input("www")
@ti.kernel
def apply_spring_force(t: ti.i32,n_springs: ti.i32):
    for i in range(n_springs):
        #print("spring", spring_anchor_a[i],spring_anchor_b[i])
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a = x[t, a]
        pos_b = x[t, b]
        dist = pos_a - pos_b
        length = dist.norm() + 1e-4
        #print("spring actuation",spring_actuation[i])
        #print("activation", act[t, i])
        #print("spring_len",spring_length[i])
        target_length = spring_length[i] * (1.0 +
                                            spring_actuation[i] * act[t, i])
        impulse = dt * (length -
                        target_length) * spring_stiffness[i] / length * dist

        #print("pos a",pos_a,"pos_b",pos_b,"dist",dist)
        #print(length, target_length, (length - target_length))
        #print("impulse", impulse)
        ti.atomic_add(v_inc[t + 1, a], -impulse)
        ti.atomic_add(v_inc[t + 1, b], impulse)


use_toi = False


@ti.kernel
def advance_toi(t: ti.i32, n_objects: ti.i32,ground_height: ti.f32):
    #print("advance",n_objects)
    #print("ground",ground_height)
    for i in range(n_objects):
        s = math.exp(-dt * damping)
        old_v = s * v[t - 1, i] + dt * gravity * ti.Vector([0.0, 1.0
                                                            ]) + v_inc[t, i]
        old_x = x[t - 1, i]
        new_x = old_x + dt * old_v
        toi = 0.0
        new_v = old_v
        if new_x[1] < ground_height and old_v[1] < -1e-4:
            toi = -(old_x[1] - ground_height) / old_v[1]
            new_v = ti.Vector([0.0, 0.0])
        new_x = old_x + toi * old_v + (dt - toi) * new_v

        v[t, i] = new_v
        x[t, i] = new_x


@ti.kernel
def advance_no_toi(t: ti.i32):
    for i in range(n_objects):
        s = math.exp(-dt * damping)
        old_v = s * v[t - 1, i] + dt * gravity * ti.Vector([0.0, 1.0
                                                            ]) + v_inc[t, i]
        old_x = x[t - 1, i]
        new_v = old_v
        depth = old_x[1] - ground_height
        if depth < 0 and new_v[1] < 0:
            # friction projection
            new_v[0] = 0
            new_v[1] = 0
        new_x = old_x + dt * new_v
        v[t, i] = new_v
        x[t, i] = new_x


@ti.kernel
def compute_loss(t: ti.i32):
    loss[None] = -x[t, head_id][0]


gui = ti.GUI("Mass Spring Robot", (512, 512), background_color=0xFFFFFF)


def forward(output=None, visualize=True):
    global use_toi
    use_toi = True

    #for i in range(n_objects):
    #    print(x[0, i][0], x[0, i][1])

    interval = vis_interval
    if output:
        interval = output_vis_interval
        os.makedirs('mass_spring/{}/'.format(output), exist_ok=True)

    total_steps = steps if not output else steps * 2
    t=0

    for t in range(1, total_steps):

        compute_center(t - 1,n_objects)

        #nn1(t - 1)
        sin_wave(t - 1,n_springs)



        apply_spring_force(t - 1,n_springs)


        if use_toi:
            advance_toi(t,n_objects,ground_height)
        else:
            advance_no_toi(t)

        if (t + 1) % interval == 0 and visualize:
            gui.line(begin=(0, ground_height),
                     end=(1, ground_height),
                     color=0x0,
                     radius=3)

            def circle(x, y, color):
                gui.circle((x, y), ti.rgb_to_hex(color), 2)

            for i in range(n_springs):

                def get_pt(x):
                    return (x[0], x[1])

                #print(get_pt(x[t, spring_anchor_a[i]]), get_pt(x[t, spring_anchor_b[i]]))
                #print((t, spring_anchor_a[i]), (t, spring_anchor_b[i]))

                a = act[t - 1, i] * 0.5
                r = 2
                if spring_actuation[i] == 0:
                    a = 0
                    c = 0x222222
                else:
                    r = 4
                    c = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))

                gui.line(begin=get_pt(x[t, spring_anchor_a[i]]),
                         end=get_pt(x[t, spring_anchor_b[i]]),
                         radius=r,
                         color=c)
            #input("step")
            for i in range(n_objects):
                color = (0.4, 0.6, 0.6)
                if i == head_id:
                    color = (0.8, 0.2, 0.3)
                circle(x[t, i][0], x[t, i][1], color)
            # circle(goal[None][0], goal[None][1], (0.6, 0.2, 0.2))

            if output:
                gui.show('mass_spring/{}/{:04d}.png'.format(output, t))
            else:
                gui.show()

    start_point = center[0][0]

    end_point = center[t-1][0]
    loss[None] = 0
    compute_loss(steps - 1)

    #print(loss[None],start_point,end_point)
    return (end_point-start_point)


@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            v_inc[t, i] = ti.Vector([0.0, 0.0])

@ti.kernel
def clear_all(n_objects: ti.i32):
    #print(n_objects)

    for t in range(0, max_steps):
        for i in range(0, n_objects):
            v_inc[t, i] = ti.Vector([0.0, 0.0])
            x[t, i] = ti.Vector([0.0, 0.0])
            v[t, i] = ti.Vector([0.0, 0.0])

def clear():

    #clear_states()
    clear_all(n_objects)



def setup_robot(objects, springs):
    global n_objects, n_springs, ground_height
    n_objects = len(objects)
    n_springs = len(springs)

    clear_all(n_objects)

    #print('n_objects=', n_objects, '   n_springs=', n_springs)
    min_height = 10
    for i in range(n_objects):
        o = objects[i]

        x[0, i][0] = np.copy(o[0])
        x[0, i][1] = np.copy(o[1])
        if x[0, i][1] < min_height:
            min_height=x[0, i][1]

    ground_height = min_height


    #print(np.shape(x))
    #input("sha")

    for i in range(n_springs):
        s = springs[i]
        #print("spring",s)
        spring_anchor_a[i] = s[0]
        spring_anchor_b[i] = s[1]
        spring_length[i] = s[2]
        spring_stiffness[i] = s[3]
        spring_actuation[i] = s[4]
        spring_phase[i] = s[5]
    #input("s")

def optimize(toi, visualize):
    global use_toi
    use_toi = toi
    for i in range(n_hidden):
        for j in range(n_input_states()):
            weights1[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + n_input_states())) * 2


    for i in range(n_springs):
        for j in range(n_hidden):
            # TODO: n_springs should be n_actuators
            weights2[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + n_springs)) * 3


    losses = []

    # forward('initial{}'.format(robot_id), visualize=visualize)
    for iter in range(100):

        clear()
        # with ti.Tape(loss) automatically clears all gradients
        with ti.Tape(loss):
            forward(visualize=visualize)

        print('Iter=', iter, 'Loss=', loss[None])

        total_norm_sqr = 0
        for i in range(n_hidden):
            for j in range(n_input_states()):
                print(weights1.grad[i, j]**2)
                total_norm_sqr += weights1.grad[i, j]**2
            total_norm_sqr += bias1.grad[i]**2

        for i in range(n_springs):
            for j in range(n_hidden):
                total_norm_sqr += weights2.grad[i, j]**2
            total_norm_sqr += bias2.grad[i]**2
        input("ejje")
        print(total_norm_sqr)
        input("errr")
        # scale = learning_rate * min(1.0, gradient_clip / total_norm_sqr ** 0.5)
        gradient_clip = 0.2
        scale = gradient_clip / (total_norm_sqr**0.5 + 1e-6)
        for i in range(n_hidden):
            for j in range(n_input_states()):
                weights1[i, j] -= scale * weights1.grad[i, j]
            bias1[i] -= scale * bias1.grad[i]

        for i in range(n_springs):
            for j in range(n_hidden):
                weights2[i, j] -= scale * weights2.grad[i, j]
            bias2[i] -= scale * bias2.grad[i]
        losses.append(loss[None])

    return losses

'''
robot_id = 0
if len(sys.argv) != 3:
    print(
        "Usage: python3 mass_spring.py [robot_id=0, 1, 2, ...] [task=train/plot]"
    )
    exit(-1)
else:
    robot_id = int(sys.argv[1])
    task = sys.argv[2]
'''
def evolve(seed=1):

    np.random.seed(seed)
    treshold = 0.1
    rob = Morph_evolve(first=True)
    #rob.create_first_morph()
    setup_robot(rob.objects, rob.springs)
    forward(visualize=False)  # optimize(toi=True, visualize=True)
    clear()
    #setup_fake_robot(fake_objects,fake_springs)

    pop_size= 5
    epochs = 100
    population = []
    fitness = -100
    mut_rate = 0.1
    for i in range(pop_size):
        rob = Morph_evolve(first=False)
        setup_robot(rob.objects, rob.springs)
        fit = forward(visualize=False)  # optimize(toi=True, visualize=True)
        #print(fit)
        if fit > fitness:
            fitness=fit
            best_ind = i
        clear()
        population.append(rob)
    best_morph = population[best_ind]


    clear()

    print("best initial fitness ",fitness)
    for i in range(epochs):


        tmp_best = None
        tmp_rate = None
        for id in range(pop_size-1):
            rob = copy(best_morph)
            rob.mutate_morph(mut_rate)
            setup_robot(rob.objects, rob.springs)
            fit = forward(visualize=False)  # optimize(toi=True, visualize=True)
            clear()
            #print(fit)
            if fit >= fitness - (fitness*treshold):
                fitness = fit
                tmp_best = rob
                tmp_rate = mut_rate*1.4
            else:
                tmp_rate = mut_rate*(1.4)**(-0.25)
        if tmp_best is not None:
            best_morph = tmp_best
        mut_rate = tmp_rate
        print(" epoch ",i, " fitness ",fitness," mut_rate ",mut_rate)
        #input("epoch ")

    setup_robot(best_morph.objects, best_morph.springs)
    fit = forward(visualize=True)  # optimize(toi=True, visualize=True)
    clear()
    #np.save(workdir+"/objectsS"+str(seed)+".npy",np.array(best_morph.objects))
    #np.save(workdir+"/springsS"+str(seed)+".npy", np.array(best_morph.springs))
    # save it
    with open('morhpS'+str(seed), 'wb') as file:
        pickle.dump(best_morph, file)

        #for rob in population:
        #    setup_robot(rob.objects, rob.springs)
        #    forward()  # optimize(toi=True, visualize=True)
        #    clear()

def run_test(seed=1):

    np.random.seed(seed)
    # load it
    with open('morhpS'+str(seed), 'rb') as file2:
        morph = pickle.load(file2)
    #rob.create_first_morph()
    setup_robot(morph.objects, morph.springs)
    forward(visualize=True)  # optimize(toi=True, visualize=True)
    clear()


def main():
    #ti.init(default_fp=real)

    rob = Morph_evolve()
    #rob.mutate_morph()
    #rob_2 = Morph_2()

    setup_robot(rob.objects,rob.springs) #setup_robot(*robots[robot_id]())

    if task == 'plot':
        ret = {}
        for toi in [False, True]:
            ret[toi] = []
            for i in range(5):
                losses = optimize(toi=toi, visualize=False)
                # losses = gaussian_filter(losses, sigma=3)
                plt.plot(losses, 'g' if toi else 'r')
                ret[toi].append(losses)

        import pickle
        pickle.dump(ret, open('losses.pkl', 'wb'))
        print("Losses saved to losses.pkl")
    else:
        forward()#optimize(toi=True, visualize=True)
        clear()

    rob.mutate_morph()
    setup_robot(rob.objects, rob.springs)
    input("rr")
    if task == 'plot':
        ret = {}
        for toi in [False, True]:
            ret[toi] = []
            for i in range(5):
                losses = optimize(toi=toi, visualize=False)
                # losses = gaussian_filter(losses, sigma=3)
                plt.plot(losses, 'g' if toi else 'r')
                ret[toi].append(losses)

        import pickle
        pickle.dump(ret, open('losses.pkl', 'wb'))
        print("Losses saved to losses.pkl")
    else:
        forward()  # optimize(toi=True, visualize=True)
        clear()
    print(x)
    input("ejeje")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fileini', help='name of ini file', type=str, default="")
    parser.add_argument('-t', '--test', help='if testing', type=bool, default=False)
    parser.add_argument('-s', '--seed', help='random generator seed', type=int, default=1)
    args = parser.parse_args()
    reset_all()
    if args.test:
        run_test(args.seed)
    else:
        evolve(args.seed)
