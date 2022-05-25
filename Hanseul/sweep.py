def make_config(classify=True, complete=True, cpl_scheme=None, mask_scheme=None, gpu=None, save=False):
    assert classify or complete
    
    if classify and complete:
        sweep_name = f'Joint Training (scheme:{cpl_scheme})'
    elif complete:
        sweep_name = f'Completion Only (scheme:{cpl_scheme})'
    else:
        sweep_name = f'Classification Only (scheme:{mask_scheme})'
    
    sweep_config = dict(
        name = sweep_name,
        method = 'grid', # grid, random, bayes
        metric = dict(
            name = 'Best/Classify/Loss' if classify else 'Best/Complete/Loss',  # Note: CrossEntropyLoss is used for Validation
            goal = 'minimize',
        ),
        parameters = dict(
            # Tuning step 1: model choice  (grid, 20 runs)
            encoder_mode=dict(values=['HYBRID_SA','HYBRID','FC','SA','ISA']), 
            num_enc_layers=dict(values=[8, 4]),
            pooler_mode=dict(values=['PMA', 'SumPool']),
            # Tuning step 2: optimization details (bayes, 200 runs)
            #num_enc_layers=dict(min=4, max=8, distribution='int_uniform'),
            #num_dec_layers=dict(min=0, max=2, distribution='int_uniform'),
            #lr=dict(min=1e-5, max=1e-3, distribution='log_uniform_values'),
            #dropout=dict(min=0., max=0.5, distribution='q_uniform', q=0.1),
            #weight_decay=dict(min=1e-4, max=1e-1, distribution='log_uniform_values'),
            #optimizer_name=dict(values=['AdamW','Adam','NesterovSGD','RMSprop','MomentumSGD','SGD']),
        )
    )
    
    fixed_config = dict(
        classify=classify,
        complete=complete,
        cpl_scheme=cpl_scheme,
        mask_scheme=mask_scheme,
        #encoder_mode = 'FC'
        #pooler_mode = 'PMA'
        #num_enc_layers = 8
        num_dec_layers=0,
        
        lr=1e-4,
        dropout=0.1,
        optimizer_name='AdamW',
        weight_decay=0.001,
        
        batch_size=64,
        dim_embedding=512,
        dim_hidden=512,
        loss='MultiClassASLoss',
        
        num_inds=10,
        step_size=5,
        step_factor=0.25,
        early_stop_patience=10,  # early stop
        data_dir='./Container',
        batch_size_eval=2048,
        n_epochs=100,
        seed=42,
        subset_length=None,  # Default: None
        verbose=True,
        gpu=gpu,
        save_model=save,
    )
    sweep_config['parameters'].update({
        k: {'value': v} for k, v in fixed_config.items()
    })
    return sweep_config

