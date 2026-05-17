use std::path::PathBuf;
use vulkano_engine::assets::asset_manager::AssetManager;

#[test]
fn test_load_3d_model_success() {
    let mut asset_manager = AssetManager::new();

    let test_model_path = "../engine/data/models/suzanne_2_material.glb";

    let model_id = asset_manager.load_model(test_model_path);

    assert!(
        model_id.id != "0",
        "Modelio ID neturėtų būti 0 arba neteisingas"
    );

    let model_data = asset_manager.get_model(&model_id);
    assert!(
        model_data.is_some(),
        "Modelis nebuvo išsaugotas AssetManager'yje!"
    );
}
