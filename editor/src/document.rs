//! The scene being edited. Entities and components live in the shipyard `World`; this
//! adds what an editor needs on top: list order, selection, the environment, and undo.

use std::collections::HashSet;
use std::path::PathBuf;

use shipyard::{EntityId, World};
use vulkano_engine::scene::scene_file::{
    ComponentData, ComponentRegistry, EntityData, Environment, FieldValue, Name, SceneFile,
};

const MAX_HISTORY: usize = 200;

pub struct Document {
    /// The scene's entities, in the order the Entities dock lists them.
    pub order: Vec<EntityId>,
    pub selection: Option<EntityId>,
    pub environment: Environment,
    pub path: Option<PathBuf>,
    /// Entities whose model just changed. If the new model has a skeleton they get an
    /// `Animator` once it has loaded.
    pub wants_animator: HashSet<EntityId>,
    current: SceneFile,
    saved: SceneFile,
    committed: SceneFile,
    undo_stack: Vec<SceneFile>,
    redo_stack: Vec<SceneFile>,
}

impl Document {
    pub fn new() -> Self {
        Self {
            order: Vec::new(),
            selection: None,
            environment: Environment::default(),
            path: None,
            wants_animator: HashSet::new(),
            current: SceneFile::default(),
            saved: SceneFile::default(),
            committed: SceneFile::default(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }

    pub fn title(&self) -> String {
        match &self.path {
            Some(path) => path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string()),
            None => "[unsaved]".to_string(),
        }
    }

    pub fn is_dirty(&self) -> bool {
        self.current != self.saved
    }

    /// Replace everything in the world with a scene from a file (or an empty one).
    /// Returns any problems found. Its models must already be loaded.
    pub fn open(
        &mut self,
        world: &mut World,
        registry: &ComponentRegistry,
        scene: SceneFile,
        path: Option<PathBuf>,
    ) -> Vec<String> {
        for entity in self.order.drain(..) {
            world.delete_entity(entity);
        }
        let warnings = self.spawn_scene(world, registry, &scene);
        self.selection = None;
        self.path = path;
        self.wants_animator.clear();
        self.refresh(world, registry);
        self.saved = self.current.clone();
        self.committed = self.current.clone();
        self.undo_stack.clear();
        self.redo_stack.clear();
        warnings
    }

    fn spawn_scene(
        &mut self,
        world: &mut World,
        registry: &ComponentRegistry,
        scene: &SceneFile,
    ) -> Vec<String> {
        let mut warnings = Vec::new();
        self.environment = scene.environment.clone();
        for data in &scene.entities {
            let (entity, problems) = registry.spawn(world, data);
            self.order.push(entity);
            warnings.extend(problems);
        }
        warnings
    }

    /// Re-read the scene from the world. Call once per frame, after any edits.
    pub fn refresh(&mut self, world: &World, registry: &ComponentRegistry) {
        self.current = SceneFile {
            environment: self.environment.clone(),
            entities: self.order.iter().map(|e| registry.capture(world, *e)).collect(),
        };
    }

    pub fn save_to(&mut self, path: PathBuf) -> Result<(), String> {
        self.current.save(&path)?;
        self.saved = self.current.clone();
        self.path = Some(path);
        Ok(())
    }

    /// Record the current state as an undo step if it changed. Passing `interacting`
    /// while a drag is in progress makes one drag a single step.
    pub fn commit_changes(&mut self, interacting: bool) {
        if !interacting && self.current != self.committed {
            let previous = std::mem::replace(&mut self.committed, self.current.clone());
            self.undo_stack.push(previous);
            if self.undo_stack.len() > MAX_HISTORY {
                self.undo_stack.remove(0);
            }
            self.redo_stack.clear();
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn undo(&mut self, world: &mut World, registry: &ComponentRegistry) {
        self.commit_changes(false);
        if let Some(previous) = self.undo_stack.pop() {
            self.redo_stack.push(self.current.clone());
            self.restore(world, registry, previous);
        }
    }

    pub fn redo(&mut self, world: &mut World, registry: &ComponentRegistry) {
        self.commit_changes(false);
        if let Some(next) = self.redo_stack.pop() {
            self.undo_stack.push(self.current.clone());
            self.restore(world, registry, next);
        }
    }

    fn restore(&mut self, world: &mut World, registry: &ComponentRegistry, scene: SceneFile) {
        // Entities are recreated, so the selection is carried over by list position.
        let selected = self.selection.and_then(|s| self.order.iter().position(|e| *e == s));
        for entity in self.order.drain(..) {
            world.delete_entity(entity);
        }
        self.spawn_scene(world, registry, &scene);
        self.selection = selected.and_then(|i| self.order.get(i).copied());
        self.refresh(world, registry);
        self.committed = self.current.clone();
    }

    fn unique_name(&self, wanted: &str, world: &World, ignore: Option<EntityId>) -> String {
        let taken = |name: &str| {
            self.order.iter().any(|e| {
                Some(*e) != ignore && world.get::<&Name>(*e).is_ok_and(|n| n.0 == name)
            })
        };
        if !taken(wanted) {
            return wanted.to_string();
        }
        let base = wanted.trim_end_matches(|c: char| c.is_ascii_digit());
        let base = if base.is_empty() { wanted } else { base };
        (2..).map(|i| format!("{base}{i}")).find(|n| !taken(n)).unwrap()
    }

    /// Create an entity with the given components and name, and select it.
    pub fn create(
        &mut self,
        world: &mut World,
        registry: &ComponentRegistry,
        name: &str,
        data: &EntityData,
    ) -> EntityId {
        let mut data = data.clone();
        data.components.retain(|c| c.component != "Name");
        let name = self.unique_name(name, world, None);
        data.components.insert(
            0,
            ComponentData::named("Name").with("name", FieldValue::Text(name)),
        );
        let (entity, _) = registry.spawn(world, &data);
        self.order.push(entity);
        self.selection = Some(entity);
        entity
    }

    pub fn delete(&mut self, world: &mut World, entity: EntityId) {
        self.order.retain(|e| *e != entity);
        if self.selection == Some(entity) {
            self.selection = None;
        }
        self.wants_animator.remove(&entity);
        world.delete_entity(entity);
    }

    pub fn duplicate(&mut self, world: &mut World, registry: &ComponentRegistry, entity: EntityId) {
        let data = registry.capture(world, entity);
        let name = data.name().unwrap_or("Entity").to_string();
        let copy = self.create(world, registry, &name, &data);
        // Keep the copy next to the original.
        self.order.pop();
        let index = self.order.iter().position(|e| *e == entity).map_or(self.order.len(), |i| i + 1);
        self.order.insert(index, copy);
    }

    pub fn display_name(&self, world: &World, entity: EntityId) -> String {
        match world.get::<&Name>(entity) {
            Ok(name) => name.0.clone(),
            Err(_) => {
                let index = self.order.iter().position(|e| *e == entity).unwrap_or(0);
                format!("Entity {}", index + 1)
            }
        }
    }

    pub fn rename(&self, world: &mut World, registry: &ComponentRegistry, entity: EntityId, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            return;
        }
        let name = self.unique_name(name, world, Some(entity));
        let def = registry.def("Name").unwrap();
        if !def.has(world, entity) {
            let _ = def.add(world, entity);
        }
        def.set_field(world, entity, "name", &FieldValue::Text(name));
    }

    /// Move an entity to just before the entity at `target` in the list, or to the end.
    pub fn reorder(&mut self, entity: EntityId, target: Option<EntityId>) {
        if target == Some(entity) {
            return;
        }
        self.order.retain(|e| *e != entity);
        let index = target
            .and_then(|t| self.order.iter().position(|e| *e == t))
            .unwrap_or(self.order.len());
        self.order.insert(index, entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vulkano_engine::scene::components::transform::Transform;

    fn setup() -> (Document, World, ComponentRegistry) {
        (Document::new(), World::new(), ComponentRegistry::with_engine_components())
    }

    fn with_transform() -> EntityData {
        EntityData { components: vec![ComponentData::named("Transform")] }
    }

    fn sync(doc: &mut Document, world: &World, registry: &ComponentRegistry, interacting: bool) {
        doc.refresh(world, registry);
        doc.commit_changes(interacting);
    }

    fn names(doc: &Document, world: &World) -> Vec<String> {
        doc.order.iter().map(|e| doc.display_name(world, *e)).collect()
    }

    #[test]
    fn created_entities_are_real_world_entities() {
        let (mut doc, mut world, registry) = setup();
        let entity = doc.create(&mut world, &registry, "Crate", &with_transform());
        assert!(world.get::<&Transform>(entity).is_ok());
        assert_eq!(world.get::<&Name>(entity).unwrap().0, "Crate");
        assert_eq!(doc.selection, Some(entity));
    }

    #[test]
    fn names_stay_unique() {
        let (mut doc, mut world, registry) = setup();
        doc.create(&mut world, &registry, "Light", &with_transform());
        doc.create(&mut world, &registry, "Light", &with_transform());
        let third = doc.create(&mut world, &registry, "Other", &with_transform());
        doc.rename(&mut world, &registry, third, "Light");
        assert_eq!(names(&doc, &world), ["Light", "Light2", "Light3"]);
    }

    #[test]
    fn deleting_removes_it_from_the_world_and_selection() {
        let (mut doc, mut world, registry) = setup();
        let entity = doc.create(&mut world, &registry, "A", &with_transform());
        doc.delete(&mut world, entity);
        assert!(doc.order.is_empty());
        assert_eq!(doc.selection, None);
        assert!(world.get::<&Transform>(entity).is_err());
    }

    #[test]
    fn duplicates_copy_components_and_sit_next_to_the_original() {
        let (mut doc, mut world, registry) = setup();
        let a = doc.create(&mut world, &registry, "A", &with_transform());
        doc.create(&mut world, &registry, "B", &with_transform());
        world.get::<&mut Transform>(a).unwrap().set_position(1.0, 2.0, 3.0);

        doc.duplicate(&mut world, &registry, a);

        assert_eq!(names(&doc, &world), ["A", "A2", "B"]);
        let copy = doc.selection.unwrap();
        assert_ne!(copy, a);
        assert_eq!(world.get::<&Transform>(copy).unwrap().get_position_vector(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn reordering_only_changes_the_list() {
        let (mut doc, mut world, registry) = setup();
        let a = doc.create(&mut world, &registry, "A", &with_transform());
        let b = doc.create(&mut world, &registry, "B", &with_transform());
        let c = doc.create(&mut world, &registry, "C", &with_transform());
        doc.reorder(c, Some(a));
        assert_eq!(names(&doc, &world), ["C", "A", "B"]);
        doc.reorder(c, None);
        assert_eq!(names(&doc, &world), ["A", "B", "C"]);
        doc.reorder(b, Some(b));
        assert_eq!(names(&doc, &world), ["A", "B", "C"]);
    }

    #[test]
    fn undo_and_redo_restore_entities_and_selection() {
        let (mut doc, mut world, registry) = setup();
        let a = doc.create(&mut world, &registry, "A", &with_transform());
        sync(&mut doc, &world, &registry, false);
        world.get::<&mut Transform>(a).unwrap().set_position(5.0, 0.0, 0.0);
        sync(&mut doc, &world, &registry, false);

        doc.undo(&mut world, &registry);
        let restored = doc.selection.unwrap();
        assert_eq!(world.get::<&Transform>(restored).unwrap().get_position_vector(), [0.0; 3]);

        doc.redo(&mut world, &registry);
        let restored = doc.selection.unwrap();
        assert_eq!(world.get::<&Transform>(restored).unwrap().get_position_vector()[0], 5.0);
    }

    #[test]
    fn undo_brings_back_a_deleted_entity_and_removed_components() {
        let (mut doc, mut world, registry) = setup();
        let a = doc.create(&mut world, &registry, "A", &with_transform());
        sync(&mut doc, &world, &registry, false);

        registry.def("Transform").unwrap().remove(&mut world, a);
        sync(&mut doc, &world, &registry, false);
        doc.undo(&mut world, &registry);
        assert!(world.get::<&Transform>(doc.order[0]).is_ok());

        let entity = doc.order[0];
        doc.delete(&mut world, entity);
        sync(&mut doc, &world, &registry, false);
        assert!(doc.order.is_empty());
        doc.undo(&mut world, &registry);
        assert_eq!(names(&doc, &world), ["A"]);
    }

    #[test]
    fn a_drag_is_a_single_undo_step() {
        let (mut doc, mut world, registry) = setup();
        let a = doc.create(&mut world, &registry, "A", &with_transform());
        sync(&mut doc, &world, &registry, false);

        for x in 1..5 {
            world.get::<&mut Transform>(a).unwrap().set_position(x as f32, 0.0, 0.0);
            sync(&mut doc, &world, &registry, true);
        }
        sync(&mut doc, &world, &registry, false);

        doc.undo(&mut world, &registry);
        let entity = doc.order[0];
        assert_eq!(world.get::<&Transform>(entity).unwrap().get_position_vector()[0], 0.0);
    }

    #[test]
    fn dirty_tracks_the_saved_state() {
        let (mut doc, mut world, registry) = setup();
        doc.refresh(&world, &registry);
        assert!(!doc.is_dirty());
        doc.create(&mut world, &registry, "A", &with_transform());
        sync(&mut doc, &world, &registry, false);
        assert!(doc.is_dirty());
        doc.undo(&mut world, &registry);
        assert!(!doc.is_dirty());
    }

    #[test]
    fn opening_a_scene_replaces_the_world_and_clears_history() {
        let (mut doc, mut world, registry) = setup();
        let old = doc.create(&mut world, &registry, "Old", &with_transform());
        sync(&mut doc, &world, &registry, false);

        let scene = SceneFile {
            environment: Environment { exposure: 2.0, ..Default::default() },
            entities: vec![with_transform(), with_transform()],
        };
        let warnings = doc.open(&mut world, &registry, scene, None);

        assert!(warnings.is_empty());
        assert_eq!(doc.order.len(), 2);
        assert!(world.get::<&Transform>(old).is_err());
        assert_eq!(doc.environment.exposure, 2.0);
        assert!(!doc.can_undo());
        assert!(!doc.is_dirty());
    }
}
