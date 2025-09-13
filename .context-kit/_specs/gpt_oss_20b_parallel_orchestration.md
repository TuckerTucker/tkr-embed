# GPT-OSS-20B Parallel Agent Orchestration Plan

**Objective**: Transform tkr-embed embedding service to gpt-oss-20b text generation service using coordinated parallel AI agents

**Strategy**: File-level isolation + Interface-driven coordination + Synchronization gates

---

## Agent Workflow Architecture

### Phase 1: Foundation Setup (Parallel - No Dependencies)
**Duration**: ~2-3 hours | **Parallelism**: 3 agents simultaneously

#### Agent A: Model Research & Compatibility
```yaml
Role: model-research
Files: .context-kit/_specs/gpt_oss_20b_analysis.md
Isolation: New files only, no conflicts
Tasks:
  - Analyze gpt-oss-20b MLX compatibility with transformers/MLX-LM
  - Test model loading with quantization on 32GB Apple Silicon
  - Document memory requirements and performance characteristics
  - Validate reasoning levels (low/medium/high) functionality
  - Create model interface specification for implementation
Deliverable: gpt_oss_20b_model_spec.py
Success Criteria: Model loads successfully with Q8 quantization
```

#### Agent B: API Data Models
```yaml
Role: api-models
Files: tkr_embed/api/models.py
Isolation: Single file ownership
Tasks:
  - Design GenerationRequest/Response Pydantic models
  - Add reasoning_level, temperature, max_tokens parameters
  - Remove embedding-specific models (EmbeddingResponse, etc.)
  - Create chat message format schemas
  - Add generation validation schemas
Deliverable: Complete API models for text generation
Success Criteria: All generation endpoints have proper request/response models
```

#### Agent C: Configuration System
```yaml
Role: configuration
Files: tkr_embed/config.py
Isolation: Single file ownership
Tasks:
  - Update configuration for gpt-oss-20b parameters
  - Add generation-specific settings (max_tokens, reasoning_levels)
  - Remove embedding configurations (embedding_dim, similarity)
  - Design model loading configuration for 21B parameters
  - Update quantization thresholds for larger model
Deliverable: Updated configuration system
Success Criteria: Config supports all gpt-oss-20b requirements
```

**Sync Point 1**: All agents complete foundation, share interface specifications

---

### Phase 2: Core Implementation (Coordinated Parallel)
**Duration**: ~4-5 hours | **Parallelism**: 2 primary + 1 support agent

#### Agent A: Model Manager (Critical Path)
```yaml
Role: model-manager
Files: tkr_embed/core/model_manager.py, tkr_embed/core/
Dependencies: Phase 1 model spec + config
Isolation: tkr_embed/core/ directory ownership
Tasks:
  - Replace OpsMMEmbeddingMLX with GPTOss20bMLX class
  - Implement text generation pipeline with reasoning levels
  - Add proper quantization for 21B parameter model
  - Optimize memory allocation for Apple Silicon Metal GPU
  - Implement streaming generation support
  - Add generation parameter validation
Deliverable: Working model manager for text generation
Success Criteria: Can generate text with all reasoning levels
Critical Path: Required for API endpoint testing
```

#### Agent B: API Endpoints (Critical Path)
```yaml
Role: api-endpoints
Files: tkr_embed/api/server.py, tkr_embed/api/
Dependencies: Phase 1 data models + config
Isolation: tkr_embed/api/ directory ownership
Tasks:
  - Implement /generate endpoint (basic completion)
  - Add /chat endpoint (conversation format)
  - Remove embedding endpoints (/embed/text, /embed/image, /embed/multimodal)
  - Remove /similarity endpoint
  - Update health/info endpoints for generation model
  - Add generation-specific error handling
Deliverable: Complete API transformation
Success Criteria: All generation endpoints functional
Parallel Until: Integration testing with Agent A
```

#### Agent C: Infrastructure Support
```yaml
Role: infrastructure
Files: tkr_embed/utils/, tkr_embed/api/error_handlers.py
Dependencies: Phase 1 config updates
Isolation: tkr_embed/utils/ directory ownership
Tasks:
  - Update memory management for 21B model size
  - Modify caching system for generation workloads
  - Update error handling for generation-specific issues
  - Adapt rate limiting for token-based costs
  - Update batch processing for variable-length outputs
Deliverable: Updated infrastructure for text generation
Success Criteria: Infrastructure supports generation workloads
Parallel: Can run alongside A & B without conflicts
```

**Sync Point 2**: Core functionality integration testing

---

### Phase 3: Optimization & Cleanup (Parallel Streams)
**Duration**: ~2-3 hours | **Parallelism**: 2 specialized agents

#### Agent A: Performance & Testing
```yaml
Role: performance-testing
Files: tests/, monitoring/, tkr_embed/core/batch_processor.py
Dependencies: Working core implementation from Phase 2
Isolation: test/ directory + performance modules
Tasks:
  - End-to-end generation testing all reasoning levels
  - Performance optimization and memory profiling
  - Update batch processing for generation throughput
  - Load testing with concurrent generation requests
  - Validate quantization performance vs accuracy
Deliverable: Validated, optimized generation system
Success Criteria: 150+ tokens/sec, <100ms latency
```

#### Agent B: Cleanup & Documentation
```yaml
Role: cleanup-docs
Files: Unused files, docs/, examples/
Dependencies: Stable core implementation
Isolation: Documentation + cleanup tasks
Tasks:
  - Remove unused embedding infrastructure
  - Clean up multimodal/image processing dependencies
  - Update API documentation for generation endpoints
  - Create generation examples and usage guides
  - Clean up imports and unused utilities
Deliverable: Clean, documented codebase
Success Criteria: No unused embedding code, complete docs
```

---

## Coordination Mechanisms

### File Ownership Matrix
```yaml
Agent Territories:
  model-research:    .context-kit/_specs/ (new files)
  api-models:        tkr_embed/api/models.py
  configuration:     tkr_embed/config.py
  model-manager:     tkr_embed/core/
  api-endpoints:     tkr_embed/api/server.py, endpoints
  infrastructure:    tkr_embed/utils/, error_handlers
  performance:       tests/, monitoring/
  cleanup:           Deletion tasks, docs/
```

### Interface Contracts
```python
# Shared specifications created in Phase 1
interface_specs/
├── model_interface.py      # Agent A → All implementation agents
├── api_contracts.py        # Agent B → Testing agents
├── config_schema.py        # Agent C → All agents
└── integration_status.yml  # All agents update progress
```

### Synchronization Gates
1. **Gate 1**: Foundation complete (interface specs ready)
2. **Gate 2**: Core implementation working (integration tests pass)
3. **Gate 3**: Performance validated (ready for production)

### Conflict Prevention Strategy
- **File Isolation**: Each agent owns specific files/directories
- **Interface-First**: Shared contracts define integration points
- **Feature Flags**: Gradual rollout during development
- **Branch Strategy**: Each agent works on feature branch, merge at sync points
- **Integration Testing**: Required at each synchronization gate

---

## Execution Commands

### Phase 1: Foundation
```bash
# Launch 3 foundation agents in parallel
claude-code task-parallel \
  --agent-1 "model-research: Analyze gpt-oss-20b MLX compatibility and create interface spec" \
  --agent-2 "api-models: Design GenerationRequest/Response models in tkr_embed/api/models.py" \
  --agent-3 "configuration: Update config.py for gpt-oss-20b parameters" \
  --sync-gate "foundation-complete"
```

### Phase 2: Core Implementation
```bash
# Launch implementation agents with dependencies
claude-code task-parallel \
  --agent-1 "model-manager: Implement GPTOss20bMLX in tkr_embed/core/" \
  --agent-2 "api-endpoints: Transform API endpoints in tkr_embed/api/" \
  --agent-3 "infrastructure: Update utils and error handling" \
  --dependencies "foundation-complete" \
  --sync-gate "core-complete"
```

### Phase 3: Optimization
```bash
# Launch optimization agents
claude-code task-parallel \
  --agent-1 "performance: Test and optimize generation performance" \
  --agent-2 "cleanup: Remove embedding code and update docs" \
  --dependencies "core-complete" \
  --sync-gate "production-ready"
```

---

## Success Metrics

### Parallelism Efficiency
- **Phase 1**: 3 agents parallel → ~60% time reduction vs sequential
- **Phase 2**: 2 critical + 1 support → ~40% time reduction
- **Phase 3**: 2 parallel streams → ~50% time reduction
- **Overall**: ~50% total implementation time savings

### Quality Gates
- **Phase 1**: All interface specs complete and compatible
- **Phase 2**: End-to-end text generation functional
- **Phase 3**: Performance targets met, clean codebase

### Performance Targets
- **Generation Speed**: 150+ tokens/second
- **Latency**: <100ms per request
- **Memory**: <50% utilization on 32GB system
- **Concurrency**: 100+ simultaneous requests

---

## Risk Mitigation

### Integration Risks
- **Mitigation**: Mandatory integration testing at each sync point
- **Rollback**: Feature flags enable gradual activation
- **Monitoring**: Shared progress dashboard tracks agent status

### Conflict Resolution
- **File Conflicts**: Prevented by ownership matrix
- **Interface Conflicts**: Resolved through shared specifications
- **Integration Issues**: Addressed at synchronization gates

### Agent Coordination
- **Communication**: Shared interface specifications
- **Status Tracking**: integration_status.yml updated by all agents
- **Deadlock Prevention**: Clear dependency graph and sync points

---

## Implementation Notes

### Agent Instructions Template
```yaml
Agent Role: {role}
Phase: {phase}
Files: {file_ownership}
Dependencies: {prerequisites}
Tasks: {detailed_task_list}
Deliverable: {specific_output}
Success Criteria: {measurable_goals}
Sync Point: {next_synchronization}
```

### Monitoring Dashboard
Track parallel progress with:
- Agent status (active/blocked/complete)
- File modification tracking
- Integration test results
- Performance benchmarks

This orchestration plan enables efficient parallel development while maintaining code quality and avoiding conflicts through careful coordination and clear ownership boundaries.