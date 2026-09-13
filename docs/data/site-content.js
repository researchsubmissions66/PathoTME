/* Editorial descriptions only. Exact measurement definitions are in panels.json. */
window.PATHOTME_CONTENT = {
  architectures: {
    vila_mil: {
      scale: 'DUAL SCALE · 5× + 10×',
      title: 'Condition the image-center queries.',
      query: 'Image-center queries', aggregation: 'Center-to-patch attention',
      note: 'Low- and high-scale TME groups update the corresponding ViLa image centers before native patch aggregation. Original text prompts are preserved.'
    },
    mgpath: {
      scale: 'MULTISCALE · 5× + 10×',
      title: 'Bring biological context to graph aggregation.',
      query: '64 image-center queries', aggregation: 'Native graph aggregation',
      note: 'The conditioner updates MGPATH’s image centers before native visual aggregation. Text encoding and transport paths stay frozen. The RN50 pairing is a PathoTME feature-context extension.'
    },
    focus: {
      scale: 'SINGLE SCALE · 20×',
      title: 'Adapt the semantic queries that focus attention.',
      query: 'Encoded semantic queries', aggregation: 'Patch selection + cross-attention',
      note: 'All 16 TME groups condition FOCUS’s high-resolution class queries before relevance ranking and final visual cross-attention. Discrete patch selection is not differentiable; final attention provides the gradient to the conditioner.'
    },
    muse: {
      scale: 'SINGLE SCALE · 10×',
      title: 'Condition class semantics before expert routing.',
      query: 'Inference class semantics', aggregation: 'Sparse expert routing',
      note: 'All 16 TME groups update the inference class semantics before MUSE’s sparse expert routing. Native experts and the visual adapter stay frozen. The training-only description retrieval bank is a separate component.'
    },
    hive_mil: {
      scale: 'HIERARCHICAL · 5× + 20×',
      title: 'Add biological context to hierarchical text nodes.',
      query: '32 hierarchical text nodes', aggregation: 'Filtering + hierarchical graph',
      note: 'Each class has four coarse and twelve fine text nodes. The conditioner updates their encoded representations before HiVE-MIL’s patch filtering and hierarchical graph. Discrete selection remains non-differentiable; prompt strings stay unchanged.'
    },
    dyko: {
      scale: 'SINGLE SCALE · 20×',
      title: 'Connect retrieved concepts to the microenvironment.',
      query: 'Two class queries', aggregation: 'Visual + semantic cross-attention',
      note: 'TME conditions DyKo’s class queries after native concept retrieval and before dual visual/semantic cross-attention. The conditioner does not change clustering, concept tensors, or discrete retrieval. Concept banks are separate from the TME measurement panels.'
    }
  },
  panels: {
    nsclc: {
      id: 'CORE62', name: 'Lung microenvironment', count: 62,
      families: [
        {name: 'Tissue architecture', detail: 'Whole-slide composition, tumor core, and margins', count: 14, color: 'violet'},
        {name: 'Cell composition', detail: 'Seven cell types, two regions, percentage and density', count: 28, color: 'teal'},
        {name: 'Spatial interactions', detail: 'Immune and stromal proximity to carcinoma', count: 16, color: 'blue'},
        {name: 'Lymphoid organization', detail: 'TLS presence and transformed counts', count: 4, color: 'pink'}
      ],
      tokens: '16 tokens · 5 low-scale + 11 high-scale groups',
      detail: 'The inherited core62 panel includes TLS measurements. Missing TLS counts use the legacy zero-count convention.'
    },
    brca: {
      id: 'MORPH64', name: 'Breast tissue morphology', count: 64,
      families: [
        {name: 'Tissue structure & morphology', detail: 'Tissue proportions, geometry, core, and margins', count: 24, color: 'violet'},
        {name: 'Cell composition', detail: 'Seven cell types, two regions, percentage and density', count: 28, color: 'teal'},
        {name: 'Spatial interactions', detail: 'Fibroblast, lymphocyte, and macrophage proximity', count: 12, color: 'blue'}
      ],
      tokens: '16 tokens · 6 low-scale + 10 high-scale groups',
      detail: 'morph64 emphasizes carcinoma and stromal architecture. TLS features are excluded from this panel by its morphology hypothesis; this does not establish that TLS is irrelevant.'
    }
  }
};
