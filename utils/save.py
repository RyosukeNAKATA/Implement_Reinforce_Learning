from IPython.display import display
from JSAnimation.IPython_display import display_animation
from matplotlib import animation

class save:
    def save_as_gif(frames):
        plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
        anim.save('MountainCar-v0.mp4')
        display(display_animation(anim, default_mode='loop'))