// The player's weapon slots: unarmed hands, and two weapon slots for melee, guns or tools.
//
//   1 hands   2 slot 1   3 slot 2   Q last one
//   Left mouse primary (swing, fire, throw)   Right mouse secondary (block, aim, drop)
//   E interact (pick up, drop)   R reload
//
// Each slot is an entity (by Name) with a script that answers to the weapon interface, so
// a new weapon is a new entity and script, nothing here changes:
//
//   fn equip(owner)            it's in hand now; owner is the player
//   fn unequip()               put it away (drop anything carried, stop blocking...)
//   fn primary(down)           left mouse went down (true) or up (false)
//   fn secondary(down)         right mouse
//   fn interact()              E
//   fn reload()                R
//
// Weapons read where the player is looking from the player's `view_origin` and
// `view_dir` (player_camera), and tell the player what they're doing with
// owner.invoke("weapon_state", "blocking", true) and so on.

export let hands = "Hands";
export let slot1 = "Wrench";
export let slot2 = "";              // a gun or utility item, later
export let reach = 2.5;             // for the "holster to carry" hint

fn on_start() {
    this.names = [this.hands, this.slot1, this.slot2];
    this.current = -1;
    this.last = 0;
    this.notice = "";
    this.notice_time = 0.0;
    this.select(0);
}

fn slot_entity(i) {
    let name = this.names[i];
    if name == "" { () } else { find(name) }
}

fn label_of(i) {
    let name = this.names[i];
    if name == "" { "—" } else { name }
}

// Put away what's in hand and take out slot i.
fn select(i) {
    if i == this.current {
        return;
    }
    let next = this.slot_entity(i);
    if next == () {
        this.say(`Slot ${i + 1} is empty`);
        return;
    }
    if this.current >= 0 {
        let held = this.slot_entity(this.current);
        if held != () {
            held.invoke("unequip");
        }
        this.last = this.current;
    }
    this.current = i;
    next.invoke("equip", me());
    this.update_hud();
}

fn active() {
    if this.current < 0 { () } else { this.slot_entity(this.current) }
}

fn on_update(dt) {
    if key_pressed("1") { this.select(0); }
    if key_pressed("2") { this.select(1); }
    if key_pressed("3") { this.select(2); }
    if key_pressed("Q") { this.select(this.last); }

    let weapon = this.active();
    if weapon != () {
        if mouse_pressed("left") { weapon.invoke("primary", true); }
        if mouse_released("left") { weapon.invoke("primary", false); }
        if mouse_pressed("right") { weapon.invoke("secondary", true); }
        if mouse_released("right") { weapon.invoke("secondary", false); }
        if key_pressed("E") { weapon.invoke("interact"); }
        if key_pressed("R") { weapon.invoke("reload"); }
    }

    // A weapon in hand can't carry things; say so when looking at something movable.
    // if this.current != 0 && this.looking_at_movable() {
    //     this.set_prompt(`Holster your weapon [1] to carry`);
    // } else if this.current != 0 {
    //     this.set_prompt("");
    // }

    this.notice_time -= dt;
    let notice = find("Notice");
    if notice != () {
        notice.visible = this.notice_time > 0.0;
    }
}

fn looking_at_movable() {
    let player = me();
    let origin = player.get_var("view_origin");
    let dir = player.get_var("view_dir");
    if origin == () || dir == () {
        return false;
    }
    let hit = raycast(origin, dir, this.reach, player);
    hit != () && hit.entity != () && hit.entity.in_group("movable")
}

fn set_prompt(text) {
    let prompt = find("Prompt");
    if prompt != () {
        prompt.set_text(text);
    }
}

// A short message in the middle of the screen.
fn say(text) {
    this.notice_time = 1.5;
    let notice = find("Notice");
    if notice != () {
        notice.set_text(text);
    }
}

fn update_hud() {
    let hud = find("WeaponHud");
    if hud == () {
        return;
    }
    let parts = [];
    for i in 0..3 {
        let label = `${i + 1} ${this.label_of(i)}`;
        parts.push(if i == this.current { `[ ${label} ]` } else { `  ${label}  ` });
    }
    hud.set_text(parts[0] + "   " + parts[1] + "   " + parts[2]);
}
