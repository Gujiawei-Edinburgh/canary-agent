use crate::error::{Result, RevisionError};
use crate::ids::RevisionId;
use crate::spec::AgentSpec;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergeResult {
    pub base: RevisionId,
    pub ours: RevisionId,
    pub theirs: RevisionId,
    pub spec: Option<AgentSpec>,
    pub conflicts: Vec<MergeConflict>,
}

impl MergeResult {
    pub fn is_clean(&self) -> bool {
        self.conflicts.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergeConflict {
    pub path: String,
    pub base: Option<Value>,
    pub ours: Option<Value>,
    pub theirs: Option<Value>,
}

pub(crate) fn merge_specs(
    base_id: RevisionId,
    ours_id: RevisionId,
    theirs_id: RevisionId,
    base: &AgentSpec,
    ours: &AgentSpec,
    theirs: &AgentSpec,
) -> Result<MergeResult> {
    let base_value = serde_json::to_value(base)
        .map_err(|error| RevisionError::Serialization(error.to_string()))?;
    let ours_value = serde_json::to_value(ours)
        .map_err(|error| RevisionError::Serialization(error.to_string()))?;
    let theirs_value = serde_json::to_value(theirs)
        .map_err(|error| RevisionError::Serialization(error.to_string()))?;
    let mut conflicts = Vec::new();
    let merged_value = merge_value(
        Some(&base_value),
        Some(&ours_value),
        Some(&theirs_value),
        "$".to_string(),
        &mut conflicts,
    );
    let spec = if conflicts.is_empty() {
        let value = merged_value.ok_or_else(|| {
            RevisionError::Serialization("merged agent spec was deleted".to_string())
        })?;
        Some(
            serde_json::from_value(value)
                .map_err(|error| RevisionError::Serialization(error.to_string()))?,
        )
    } else {
        None
    };
    Ok(MergeResult {
        base: base_id,
        ours: ours_id,
        theirs: theirs_id,
        spec,
        conflicts,
    })
}

pub(crate) fn choose_merge_base(
    ours: &RevisionId,
    theirs: &RevisionId,
    ancestors: &BTreeMap<RevisionId, BTreeMap<RevisionId, usize>>,
) -> Option<RevisionId> {
    let ours_ancestors = ancestors.get(ours)?;
    let theirs_ancestors = ancestors.get(theirs)?;
    ours_ancestors
        .iter()
        .filter_map(|(id, ours_distance)| {
            theirs_ancestors
                .get(id)
                .map(|theirs_distance| ((*ours_distance + *theirs_distance), id.clone()))
        })
        .min_by(|left, right| left.cmp(right))
        .map(|(_, id)| id)
}

pub(crate) fn collect_ancestors<F>(
    start: &RevisionId,
    mut parents: F,
) -> Result<BTreeMap<RevisionId, usize>>
where
    F: FnMut(&RevisionId) -> Result<Vec<RevisionId>>,
{
    let mut distances = BTreeMap::new();
    let mut queue = VecDeque::from([(start.clone(), 0usize)]);
    while let Some((id, distance)) = queue.pop_front() {
        if distances.contains_key(&id) {
            continue;
        }
        distances.insert(id.clone(), distance);
        for parent in parents(&id)? {
            queue.push_back((parent, distance.saturating_add(1)));
        }
    }
    Ok(distances)
}

fn merge_value(
    base: Option<&Value>,
    ours: Option<&Value>,
    theirs: Option<&Value>,
    path: String,
    conflicts: &mut Vec<MergeConflict>,
) -> Option<Value> {
    if ours == theirs {
        return ours.cloned();
    }
    if ours == base {
        return theirs.cloned();
    }
    if theirs == base {
        return ours.cloned();
    }

    if let (Some(Value::Object(base)), Some(Value::Object(ours)), Some(Value::Object(theirs))) =
        (base, ours, theirs)
    {
        let keys = base
            .keys()
            .chain(ours.keys())
            .chain(theirs.keys())
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut merged = Map::new();
        for key in keys {
            let child = merge_value(
                base.get(&key),
                ours.get(&key),
                theirs.get(&key),
                format!("{path}.{key}"),
                conflicts,
            );
            if let Some(value) = child {
                merged.insert(key, value);
            }
        }
        return Some(Value::Object(merged));
    }

    conflicts.push(MergeConflict {
        path,
        base: base.cloned(),
        ours: ours.cloned(),
        theirs: theirs.cloned(),
    });
    None
}
