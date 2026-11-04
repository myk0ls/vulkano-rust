//! The scene being edited: its node tree, selection, and undo history.

use std::collections::HashSet;
use std::path::PathBuf;

use nalgebra_glm as glm;
use vulkano_engine::scene::scene_file::{
    NodeId, NodeKind, NodeTransform, SceneFile, SceneNode, WorldPose, euler_degrees_from_matrix,
};

const MAX_HISTORY: usize = 200;

/// Where a node ends up relative to the node it was dropped on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DropPlace {
    Before(NodeId),
    After(NodeId),
    ChildOf(NodeId),
    Root,
}

pub struct Document {
    pub scene: SceneFile,
    pub path: Option<PathBuf>,
    pub selection: Option<NodeId>,
    saved: SceneFile,
    committed: SceneFile,
    undo_stack: Vec<SceneFile>,
    redo_stack: Vec<SceneFile>,
}

impl Document {
    pub fn new() -> Self {
        Self::from_scene(SceneFile::default(), None)
    }

    pub fn from_scene(scene: SceneFile, path: Option<PathBuf>) -> Self {
        Self {
            saved: scene.clone(),
            committed: scene.clone(),
            scene,
            path,
            selection: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }

    pub fn is_dirty(&self) -> bool {
        self.scene != self.saved
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

    pub fn save_to(&mut self, path: PathBuf) -> Result<(), String> {
        self.scene.save(&path)?;
        self.saved = self.scene.clone();
        self.path = Some(path);
        Ok(())
    }

    /// Record the current state as an undo step if it changed. Call once per frame,
    /// passing `interacting` while a drag is in progress so one drag is one step.
    pub fn commit_changes(&mut self, interacting: bool) {
        if !interacting && self.scene != self.committed {
            let previous = std::mem::replace(&mut self.committed, self.scene.clone());
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

    pub fn undo(&mut self) {
        self.commit_changes(false);
        if let Some(previous) = self.undo_stack.pop() {
            self.redo_stack.push(std::mem::replace(&mut self.scene, previous));
            self.committed = self.scene.clone();
            self.fix_selection();
        }
    }

    pub fn redo(&mut self) {
        self.commit_changes(false);
        if let Some(next) = self.redo_stack.pop() {
            self.undo_stack.push(std::mem::replace(&mut self.scene, next));
            self.committed = self.scene.clone();
            self.fix_selection();
        }
    }

    fn fix_selection(&mut self) {
        if let Some(id) = self.selection {
            if self.node(id).is_none() {
                self.selection = None;
            }
        }
    }

    pub fn node(&self, id: NodeId) -> Option<&SceneNode> {
        self.scene.nodes.iter().find(|n| n.id == id)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut SceneNode> {
        self.scene.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn children(&self, parent: Option<NodeId>) -> impl Iterator<Item = &SceneNode> {
        self.scene.nodes.iter().filter(move |n| n.parent == parent)
    }

    fn descendants(&self, id: NodeId) -> HashSet<NodeId> {
        let mut found = HashSet::from([id]);
        // Parents come before children, so one pass reaches every descendant.
        for node in &self.scene.nodes {
            if node.parent.is_some_and(|p| found.contains(&p)) {
                found.insert(node.id);
            }
        }
        found
    }

    fn next_id(&self) -> NodeId {
        self.scene.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1
    }

    fn unique_name(&self, parent: Option<NodeId>, wanted: &str, ignore: Option<NodeId>) -> String {
        let taken = |name: &str| {
            self.children(parent)
                .any(|n| Some(n.id) != ignore && n.name == name)
        };
        if !taken(wanted) {
            return wanted.to_string();
        }
        let base = wanted.trim_end_matches(|c: char| c.is_ascii_digit());
        let base = if base.is_empty() { wanted } else { base };
        (2..)
            .map(|i| format!("{base}{i}"))
            .find(|name| !taken(name))
            .unwrap()
    }

    /// Add a node as the last child of `parent`, select it, and return its id.
    pub fn add_node(&mut self, parent: Option<NodeId>, name: &str, kind: NodeKind) -> NodeId {
        self.add_node_at(parent, name, kind, NodeTransform::default())
    }

    pub fn add_node_at(
        &mut self,
        parent: Option<NodeId>,
        name: &str,
        kind: NodeKind,
        transform: NodeTransform,
    ) -> NodeId {
        let id = self.next_id();
        let node = SceneNode {
            id,
            name: self.unique_name(parent, name, None),
            parent,
            transform,
            visible: true,
            kind,
            physics: None,
        };
        let index = match parent {
            Some(p) => self.subtree_end(p),
            None => self.scene.nodes.len(),
        };
        self.scene.nodes.insert(index, node);
        self.selection = Some(id);
        id
    }

    /// Index just past the last node of the subtree rooted at `id`.
    fn subtree_end(&self, id: NodeId) -> usize {
        let members = self.descendants(id);
        self.scene
            .nodes
            .iter()
            .rposition(|n| members.contains(&n.id))
            .map_or(self.scene.nodes.len(), |i| i + 1)
    }

    pub fn remove_subtree(&mut self, id: NodeId) {
        let members = self.descendants(id);
        self.scene.nodes.retain(|n| !members.contains(&n.id));
        self.fix_selection();
    }

    /// Copy a node and everything under it, placing the copy right after the original.
    pub fn duplicate(&mut self, id: NodeId) -> Option<NodeId> {
        let parent = self.node(id)?.parent;
        let members = self.descendants(id);
        let mut next = self.next_id();
        let mut remap = std::collections::HashMap::new();
        for node in self.scene.nodes.iter().filter(|n| members.contains(&n.id)) {
            remap.insert(node.id, next);
            next += 1;
        }
        let name = self.unique_name(parent, &self.node(id)?.name.clone(), None);
        let copies: Vec<SceneNode> = self
            .scene
            .nodes
            .iter()
            .filter(|n| members.contains(&n.id))
            .map(|n| {
                let mut copy = n.clone();
                copy.id = remap[&n.id];
                copy.parent = n.parent.map(|p| remap.get(&p).copied().unwrap_or(p));
                if n.id == id {
                    copy.name = name.clone();
                }
                copy
            })
            .collect();
        let index = self.subtree_end(id);
        self.scene.nodes.splice(index..index, copies);
        self.selection = Some(remap[&id]);
        Some(remap[&id])
    }

    pub fn rename(&mut self, id: NodeId, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            return;
        }
        let Some(parent) = self.node(id).map(|n| n.parent) else {
            return;
        };
        let name = self.unique_name(parent, name, Some(id));
        if let Some(node) = self.node_mut(id) {
            node.name = name;
        }
    }

    /// Move a node in the tree, keeping where it is in the world. Returns false for
    /// moves that aren't possible, such as dropping a node into itself.
    pub fn move_node(&mut self, id: NodeId, place: DropPlace) -> bool {
        let members = self.descendants(id);
        let target = match place {
            DropPlace::Before(t) | DropPlace::After(t) | DropPlace::ChildOf(t) => Some(t),
            DropPlace::Root => None,
        };
        if target.is_some_and(|t| members.contains(&t)) {
            return false;
        }
        let new_parent = match place {
            DropPlace::ChildOf(t) => Some(t),
            DropPlace::Before(t) | DropPlace::After(t) => match self.node(t) {
                Some(node) => node.parent,
                None => return false,
            },
            DropPlace::Root => None,
        };

        let poses = self.scene.world_poses();
        let world = poses[&id];
        let parent_pose = new_parent.map_or(WorldPose::IDENTITY, |p| poses[&p]);
        let local = local_transform_from_world(&world, &parent_pose);

        let mut moved: Vec<SceneNode> = Vec::new();
        self.scene.nodes.retain(|n| {
            let is_member = members.contains(&n.id);
            if is_member {
                moved.push(n.clone());
            }
            !is_member
        });
        for node in &mut moved {
            if node.id == id {
                node.parent = new_parent;
                node.transform = local.clone();
            }
        }
        let index = match place {
            DropPlace::Before(t) => self.scene.nodes.iter().position(|n| n.id == t).unwrap(),
            DropPlace::After(t) | DropPlace::ChildOf(t) => self.subtree_end(t),
            DropPlace::Root => self.scene.nodes.len(),
        };
        self.scene.nodes.splice(index..index, moved);

        let wanted = self.node(id).unwrap().name.clone();
        let name = self.unique_name(new_parent, &wanted, Some(id));
        self.node_mut(id).unwrap().name = name;
        true
    }

    /// Set a node's transform from a world-space pose.
    pub fn set_world_pose(&mut self, id: NodeId, world: &WorldPose) {
        let poses = self.scene.world_poses();
        let parent_pose = self
            .node(id)
            .and_then(|n| n.parent)
            .and_then(|p| poses.get(&p).copied())
            .unwrap_or(WorldPose::IDENTITY);
        let local = local_transform_from_world(world, &parent_pose);
        if let Some(node) = self.node_mut(id) {
            node.transform = local;
        }
    }
}

/// The local transform that places a child at `world` under a parent at `parent`.
pub fn local_transform_from_world(world: &WorldPose, parent: &WorldPose) -> NodeTransform {
    let inverse_rotation = parent.rotation.transpose();
    let scale = if parent.scale.abs() > 1e-6 { parent.scale } else { 1.0 };
    let delta = world.position - parent.position;
    let local = inverse_rotation * glm::vec4(delta.x, delta.y, delta.z, 0.0) / scale;
    NodeTransform {
        position: [local.x, local.y, local.z],
        rotation: euler_degrees_from_matrix(&(inverse_rotation * world.rotation)),
        scale: world.scale / scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(doc: &Document) -> Vec<&str> {
        doc.scene.nodes.iter().map(|n| n.name.as_str()).collect()
    }

    fn sample() -> (Document, NodeId, NodeId, NodeId) {
        let mut doc = Document::new();
        let a = doc.add_node(None, "A", NodeKind::Empty);
        let b = doc.add_node(Some(a), "B", NodeKind::Empty);
        let c = doc.add_node(None, "C", NodeKind::Empty);
        (doc, a, b, c)
    }

    #[test]
    fn sibling_names_stay_unique() {
        let mut doc = Document::new();
        doc.add_node(None, "Light", NodeKind::Empty);
        let second = doc.add_node(None, "Light", NodeKind::Empty);
        assert_eq!(doc.node(second).unwrap().name, "Light2");
        // The same name is fine under a different parent.
        let child = doc.add_node(Some(second), "Light", NodeKind::Empty);
        assert_eq!(doc.node(child).unwrap().name, "Light");
    }

    #[test]
    fn children_are_stored_after_parents_and_inside_their_subtree() {
        let (mut doc, a, _, _) = sample();
        doc.add_node(Some(a), "D", NodeKind::Empty);
        assert_eq!(names(&doc), ["A", "B", "D", "C"]);
    }

    #[test]
    fn removing_a_node_removes_its_subtree_and_selection() {
        let (mut doc, a, b, _) = sample();
        doc.selection = Some(b);
        doc.remove_subtree(a);
        assert_eq!(names(&doc), ["C"]);
        assert_eq!(doc.selection, None);
    }

    #[test]
    fn duplicate_copies_the_whole_subtree_with_new_ids() {
        let (mut doc, a, b, _) = sample();
        let copy = doc.duplicate(a).unwrap();
        assert_eq!(names(&doc), ["A", "B", "A2", "B", "C"]);
        let child = doc.scene.nodes.iter().find(|n| n.parent == Some(copy)).unwrap();
        assert_ne!(child.id, b);
        assert_eq!(doc.selection, Some(copy));
    }

    #[test]
    fn a_node_cannot_be_moved_into_its_own_subtree() {
        let (mut doc, a, b, _) = sample();
        assert!(!doc.move_node(a, DropPlace::ChildOf(b)));
        assert!(!doc.move_node(a, DropPlace::ChildOf(a)));
        assert_eq!(names(&doc), ["A", "B", "C"]);
    }

    #[test]
    fn moves_keep_tree_order_valid() {
        let (mut doc, a, b, c) = sample();
        assert!(doc.move_node(a, DropPlace::ChildOf(c)));
        assert_eq!(names(&doc), ["C", "A", "B"]);
        assert_eq!(doc.node(a).unwrap().parent, Some(c));
        assert_eq!(doc.node(b).unwrap().parent, Some(a));

        assert!(doc.move_node(a, DropPlace::Before(c)));
        assert_eq!(names(&doc), ["A", "B", "C"]);
        assert_eq!(doc.node(a).unwrap().parent, None);
    }

    #[test]
    fn reparenting_keeps_the_world_pose() {
        let (mut doc, a, _, c) = sample();
        doc.node_mut(c).unwrap().transform = NodeTransform {
            position: [3.0, 1.0, 2.0],
            rotation: [0.0, 90.0, 0.0],
            scale: 2.0,
        };
        doc.node_mut(a).unwrap().transform.position = [1.0, 1.0, 1.0];
        let before = doc.scene.world_poses()[&a];

        assert!(doc.move_node(a, DropPlace::ChildOf(c)));

        let after = doc.scene.world_poses()[&a];
        assert!((before.matrix() - after.matrix()).norm() < 1e-4);
    }

    #[test]
    fn undo_and_redo_restore_committed_states() {
        let mut doc = Document::new();
        doc.add_node(None, "A", NodeKind::Empty);
        doc.commit_changes(false);
        doc.add_node(None, "B", NodeKind::Empty);
        doc.commit_changes(false);

        doc.undo();
        assert_eq!(names(&doc), ["A"]);
        doc.undo();
        assert!(names(&doc).is_empty());
        assert!(!doc.can_undo());

        doc.redo();
        doc.redo();
        assert_eq!(names(&doc), ["A", "B"]);
        assert!(!doc.can_redo());
    }

    #[test]
    fn a_drag_is_a_single_undo_step() {
        let mut doc = Document::new();
        let a = doc.add_node(None, "A", NodeKind::Empty);
        doc.commit_changes(false);

        for x in 1..5 {
            doc.node_mut(a).unwrap().transform.position[0] = x as f32;
            doc.commit_changes(true);
        }
        doc.commit_changes(false);

        doc.undo();
        assert_eq!(doc.node(a).unwrap().transform.position[0], 0.0);
    }

    #[test]
    fn a_new_edit_clears_redo() {
        let mut doc = Document::new();
        doc.add_node(None, "A", NodeKind::Empty);
        doc.commit_changes(false);
        doc.undo();
        doc.add_node(None, "B", NodeKind::Empty);
        doc.commit_changes(false);
        assert!(!doc.can_redo());
    }

    #[test]
    fn dirty_tracks_the_saved_state() {
        let mut doc = Document::new();
        assert!(!doc.is_dirty());
        doc.add_node(None, "A", NodeKind::Empty);
        assert!(doc.is_dirty());
        doc.undo();
        doc.commit_changes(false);
        doc.undo();
        assert!(!doc.is_dirty());
    }
}
