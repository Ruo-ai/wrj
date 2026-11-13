"""Development-only replay window sizing; leaves world coordinates unchanged."""


def fit_size(width, height, screen_width, screen_height):
    scale=min(1.,max(1,screen_width-100)/width,max(1,screen_height-140)/height)
    return max(1,int(width*scale)),max(1,int(height*scale))


def configure_replay_window():
    import arcade
    if getattr(arcade.Window,'_red_fitted_replay',False):
        return
    original=arcade.Window.run

    def fitted_run(window,*args,**kwargs):
        screen=window.screen
        width,height=fit_size(window.width,window.height,screen.width,screen.height)
        window.set_size(width,height)
        window.set_location(max(0,(screen.width-width)//2),max(0,(screen.height-height)//2-30))
        original_draw=window.on_draw

        def fitted_draw():
            # The original replay sets projection to the full world extent.
            # Resize its screen framebuffer as well, including HiDPI pixels.
            window.ctx.screen.viewport=(0,0,*window.get_framebuffer_size())
            original_draw()

        window.on_draw=fitted_draw
        print(f'Replay window fitted: {width} x {height}; world extent preserved',flush=True)
        return original(window,*args,**kwargs)

    arcade.Window.run=fitted_run
    arcade.Window._red_fitted_replay=True
