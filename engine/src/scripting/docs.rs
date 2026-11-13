//! The scripting API's documentation: shown in the editor's help and used by the
//! language server (`vulkano-rhai-lsp`) for completion, hover, and signature help.
//!
//! Each entry is an example call and what it does. The language server finds functions by
//! the names in the examples (`name(`), and treats `e.name(` as a method on an Entity.

/// Sections of (example, description) pairs.
pub const API_HELP: &[(&str, &[(&str, &str)])] = &[
    (
        "Called by the engine",
        &[
            ("fn on_start()", "Runs once when the game starts or the entity appears."),
            ("fn on_update(dt)", "Runs every frame. dt is in seconds."),
            ("fn on_late_update(dt)", "Runs every frame after physics has moved things, so a camera can follow a character without lagging behind."),
            ("fn on_collision(other)", "Something touched this entity's collider. other is an Entity."),
            ("fn on_collision_end(other)", "It stopped touching."),
            ("fn on_pressed()", "This UI button was clicked."),
            ("fn on_<signal>(...)", "Any signal this entity emits also calls this on its own scripts."),
        ],
    ),
    (
        "Values",
        &[
            ("export let name = 1.0;", "Show a value in the Inspector: numbers, true/false, text, vec3(x, y, z)."),
            ("this.name", "An exported value, or anything else this script keeps between frames."),
            ("set_global(\"score\", 10)", "A value every script can read with get_global(\"score\"). Kept across scenes."),
            ("get_global(\"score\", 0)", "With a default for when it isn't set yet."),
            ("has_global(\"score\")", "Whether a global value has been set."),
        ],
    ),
    (
        "This entity (shorthand)",
        &[
            ("position(), set_position(x, y, z)", "This entity's position; set_position also takes a vec3."),
            ("translate(x, y, z), rotate_y(deg)", "Move or turn this entity. Also rotate_x, rotate_z; translate takes a vec3 too."),
            ("set_scale(2.0)", "This entity's uniform scale."),
            ("entity_name()", "This entity's Name."),
            ("set_light_intensity(5.0), set_light_color(r, g, b)", "This entity's Pointlight."),
        ],
    ),
    (
        "Entities",
        &[
            ("me()", "This script's entity."),
            ("find(\"Name\")", "An entity by its Name, or () if there is none."),
            ("position_of(\"Name\")", "Where the entity with this Name is, or () if there is none."),
            ("get_group(\"enemies\")", "Every entity in a group, as an array."),
            ("entities_with(\"Pointlight\")", "Every entity with a component."),
            ("instantiate(\"prefabs/x.vscene\", pos)", "Spawn a prefab, offset by pos. Returns its first entity."),
            ("e.destroy()", "Remove at the end of this frame."),
            ("e.name, e.position, e.rotation, e.scale, e.visible", "Read or assign. Rotation is in degrees."),
            ("e.velocity", "Read or assign a physics body's velocity."),
            ("e.translate(v), e.rotate_y(deg)", "Move or turn by an amount. Also rotate_x, rotate_z."),
            ("e.look_at(point or entity), e.forward()", "Face something; the direction it faces."),
            ("e.distance_to(other)", "Distance to an entity or point."),
            ("e.apply_impulse(v)", "Push a physics body."),
            ("e.move_velocity = vec3(x, 0, z)", "Walk a KinematicCharacter at this speed in m/s; Y is ignored. Physics keeps it on the ground and out of walls."),
            ("e.jump(6.0), e.grounded", "Launch a KinematicCharacter upward at this speed; whether it stands on something."),
            ("e.play_animation(\"Walk\")", "On an Animator."),
            ("e.get(\"Comp\", \"field\"), e.set(...)", "Any field of any component, as in the Inspector."),
            ("e.has(\"Comp\"), e.add(...), e.remove(...)", "Check, add, or remove a component."),
            ("e.get_var(\"hp\"), e.set_var(\"hp\", 3)", "Another entity's script values."),
            ("e.set_text(\"Score: 3\")", "On a UiElement."),
            ("e.valid", "False once destroyed."),
        ],
    ),
    (
        "Signals and groups",
        &[
            ("emit(\"died\", args...)", "Send a signal from this entity, with up to 3 values."),
            ("connect(entity, \"died\", \"on_enemy_died\")", "Call a function here when entity emits died. It gets the values, then the sender."),
            ("disconnect(entity, \"died\")", "Stop listening."),
            ("add_to_group(\"enemies\")", "Join a group; also e.add_to_group. Groups can be set in the Inspector."),
            ("remove_from_group(\"enemies\"), is_in_group(...)", "Leave or check; e.in_group(...) for others."),
            ("call_group(\"enemies\", \"alert\", args...)", "Call a function on every script in a group."),
            ("e.invoke(\"open\", args...)", "Call a function on another entity's scripts, with up to 2 values. (Rhai reserves e.call for function pointers.)"),
        ],
    ),
    (
        "Time",
        &[
            ("after(2.0, \"explode\")", "Call a function of this script once, later."),
            ("every(0.5, \"shoot\")", "Call it repeatedly."),
            ("cancel_timers()", "Stop this script's timers."),
            ("time(), frame()", "Seconds and frames since the game started."),
            ("fps(), frame_ms()", "Frames per second and frame time, averaged over half a second. For a counter on a UiElement: me().set_text(`${fps()} FPS`)"),
        ],
    ),
    (
        "Input",
        &[
            ("key_down(\"W\")", "Held. Names as SDL gives them: \"Space\", \"Left\", \"Return\", ..."),
            ("key_pressed(\"Space\"), key_released(...)", "Only on the frame it went down or up."),
            ("mouse_down(\"left\"), mouse_pressed(...)", "Mouse buttons: left, right, middle. Also mouse_released."),
            ("mouse_position(), mouse_delta()", "In pixels, as vec3 with z = 0."),
            ("set_mouse_captured(false)", "Free the cursor to click UI; true gives it back to the camera."),
        ],
    ),
    (
        "Physics",
        &[
            ("raycast(origin, direction, max)", "First collider hit: a map with entity, position, normal, distance; or ()."),
            ("raycast(origin, direction, max, me())", "The same, ignoring an entity."),
            ("raycast_mask(origin, direction, max, mask)", "Only hit colliders on layers in mask, such as layer(1) | layer(3)."),
            ("e.collision_layer, e.collision_mask", "Read or assign a collider's layers as bits; layer(n) is the bit for layer n."),
        ],
    ),
    (
        "Audio",
        &[
            ("play_audio(), e.play_audio()", "Play this entity's AudioPlayback or AudioPlayback3D: from the start if stopped, carrying on if paused."),
            ("play_audio(\"data/audio/x.ogg\")", "Switch to another sound and play it from the start. Also e.play_audio(path)."),
            ("stop_audio(), e.stop_audio()", "Stop and go back to the start."),
            ("pause_audio(), resume_audio()", "Hold the position and carry on from it. Also e.pause_audio(), e.resume_audio()."),
            ("restart_audio(), e.restart_audio()", "Play from the start (the end when reversed), whatever it was doing."),
            ("reverse_audio(), e.reverse_audio()", "Turn around: a playing sound carries on backwards from where it is."),
            ("seek_audio(1.5), e.seek_audio(1.5)", "Jump to a time in seconds while playing or paused."),
            ("is_audio_playing(), audio_time(), audio_length()", "This entity's sound: playing?, seconds in, and its length."),
            ("e.audio_playing, e.audio_paused, e.audio_length", "Read another entity's sound."),
            ("e.audio_time, e.audio_reversed", "Read or assign: assigning audio_time seeks."),
            ("e.set(\"AudioPlayback\", \"volume\", 0.5)", "Change any setting: volume, pitch, looping, pan; on AudioPlayback3D min_distance, max_distance, rolloff."),
            ("fn on_audio_finished()", "A sound that doesn't loop reached its end. It's also the \"audio_finished\" signal."),
            ("play_sound(\"data/audio/coin.wav\", volume)", "Play a sound once, without an entity. The volume is optional."),
            ("play_sound_at(\"data/audio/boom.wav\", pos, volume)", "The same, heard from a point."),
            ("set_master_volume(0.5)", "The volume of everything."),
        ],
    ),
    (
        "Scenes and the game",
        &[
            ("change_scene(\"scenes/level2.vscene\")", "Load another scene after this frame."),
            ("reload_scene()", "Restart the current scene."),
            ("quit()", "Close the game."),
            ("set_post_param(\"vignette\", \"strength\", 0.8)", "Change a post effect's value."),
            ("e.set_material_param(\"glow\", 2.0)", "Change one of e's material shader @params."),
            ("set_post_enabled(\"vignette\", false)", "Turn a post effect on or off."),
        ],
    ),
    (
        "Math",
        &[
            ("vec3(x, y, z)", "With + - * /, .length(), .normalized(), .dot(), .cross(), .distance_to(), .direction_to(), .lerp()."),
            ("lerp(a, b, t), clamp(x, lo, hi)", "Also sign, smoothstep, move_toward, deg_to_rad, rad_to_deg."),
            ("rand(), rand_range(lo, hi), rand_int(lo, hi)", "Random numbers; pick(array) for a random item."),
            ("sin, cos, sqrt, abs, floor, min, max, ...", "Rhai's own math functions."),
            ("print(value), warn(text), error(text)", "Write to the Output dock."),
        ],
    ),
];
