# bpy-free tests for splat_state. Run: python3 tests/test_splat_state.py

import importlib.util
import os

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    'splat_state', os.path.join(_root, 'pobim_splats', 'splat_state.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SplatState = _mod.SplatState
EditHistory = _mod.EditHistory
State = _mod.State
serialize_rows = _mod.serialize_rows
deserialize_rows = _mod.deserialize_rows


def test_selection():
    n = 100
    st = SplatState(n)
    assert st.version == 0 and st.num_selected == 0

    # set: returns the changed indices, bumps version
    idx = np.array([1, 2, 3, 10, 50])
    ch = st.select_indices(idx, 'set')
    assert set(ch.tolist()) == set(idx.tolist())
    assert st.num_selected == 5 and st.version == 1

    # a no-op mutation returns empty and does NOT bump version
    ch = st.select_indices(idx, 'add')
    assert ch.size == 0 and st.version == 1

    ch = st.select_indices([4], 'add')
    assert ch.tolist() == [4] and st.num_selected == 6 and st.version == 2

    ch = st.select_indices([1, 2], 'remove')
    assert set(ch.tolist()) == {1, 2} and st.num_selected == 4

    # set replaces: deselect the old, select the new — both reported
    ch = st.select_indices([7, 8], 'set')
    assert set(ch.tolist()) == {3, 4, 10, 50, 7, 8}
    assert st.num_selected == 2

    st.select_all()
    assert st.num_selected == n
    st.select_none()
    assert st.num_selected == 0
    st.select_indices([0, 1, 2], 'set')
    st.select_invert()
    assert st.num_selected == n - 3


def test_visibility_and_delete():
    n = 100
    st = SplatState(n)

    st.select_indices([5, 6, 7], 'set')
    ch = st.hide_selected()
    assert set(ch.tolist()) == {5, 6, 7} and st.num_hidden == 3
    vm = st.visible_mask()
    assert not vm[5] and vm[0]
    st.unhide_all()
    assert st.num_hidden == 0

    # delete soft-deletes and clears the SELECTED bit
    st.select_none()
    st.select_indices([9, 11], 'set')
    ch = st.delete_selected()
    assert set(ch.tolist()) == {9, 11}
    assert st.num_deleted == 2 and st.num_selected == 0
    km = st.keep_mask()
    assert not km[9] and not km[11] and km[0]


def test_serialize_header_and_mismatch():
    import base64
    import zlib

    st = SplatState(123)
    st.select_indices([0, 7, 122], 'set')
    s = st.serialize()

    # payload starts with the count as an 8-byte LE uint64, then zlib data
    payload = base64.b64decode(s)
    assert int.from_bytes(payload[:8], 'little') == 123
    zlib.decompress(payload[8:])   # must be a valid zlib blob

    # roundtrip with the matching count still works
    assert np.array_equal(SplatState.deserialize(s, 123).flags, st.flags)

    # stale state (Max Splats changed / file re-pointed) must raise instead
    # of silently decoding garbage flags into the export mask
    for wrong in (122, 124, 0, 1000):
        try:
            SplatState.deserialize(s, wrong)
        except ValueError:
            pass
        else:
            raise AssertionError(f'count mismatch {wrong} did not raise')

    # corrupt payloads raise ValueError too (never other surprises)
    good_count = (123).to_bytes(8, 'little')
    for bad in ('!!!not-base64!!!',
                base64.b64encode(b'\x01\x02').decode(),           # truncated
                base64.b64encode(good_count + b'junk').decode(),  # bad zlib
                base64.b64encode(good_count + zlib.compress(b'\x00')).decode()):
        try:
            SplatState.deserialize(bad, 123)
        except ValueError:
            pass
        else:
            raise AssertionError(f'corrupt payload {bad!r} did not raise')


def test_serialize_roundtrip():
    # mixed flags across all three planes
    st = SplatState(1000)
    st.select_indices(np.arange(0, 1000, 3), 'set')
    st.hide_selected()
    st.select_indices(np.arange(1, 1000, 7), 'set')
    st.delete_selected()

    s = st.serialize()
    assert isinstance(s, str)
    back = SplatState.deserialize(s, 1000)
    assert np.array_equal(st.flags, back.flags)

    # empty cloud roundtrips cleanly
    st0 = SplatState(0)
    assert np.array_equal(SplatState.deserialize(st0.serialize(), 0).flags,
                          st0.flags)

    # a count that is not a byte multiple (packbits padding)
    st5 = SplatState(5)
    st5.select_indices([0, 4], 'set')
    st5.select_indices([2], 'add')
    st5.hide_selected()
    assert np.array_equal(
        SplatState.deserialize(st5.serialize(), 5).flags, st5.flags)


def test_edit_history():
    hist = EditHistory()
    state = SplatState(20)

    def record(label, fn):
        before = state.flags.copy()
        changed = fn()
        after = state.flags.copy()
        hist.push({'label': label,
                   'indices': changed,
                   'before': before[changed].copy(),
                   'after': after[changed].copy()})

    record('select', lambda: state.select_indices([1, 2, 3], 'set'))
    assert state.num_selected == 3
    record('hide', lambda: state.hide_selected())
    assert state.num_hidden == 3

    # undo chain
    assert hist.can_undo
    hist.undo(state)                 # undo hide
    assert state.num_hidden == 0 and state.num_selected == 3
    hist.undo(state)                 # undo select
    assert state.num_selected == 0 and not hist.can_undo

    # redo chain
    assert hist.can_redo
    hist.redo(state)                 # redo select
    assert state.num_selected == 3
    hist.redo(state)                 # redo hide
    assert state.num_hidden == 3 and not hist.can_redo

    # redo invalidation: after an undo, a fresh push clears the redo tail
    hist.undo(state)                 # back to selected, hide undone
    assert hist.can_redo
    record('delete', lambda: state.delete_selected())
    assert not hist.can_redo
    assert state.num_deleted == 3 and state.num_selected == 0

    hist.clear()
    assert not hist.can_undo and not hist.can_redo


def test_edit_history_cap():
    # bounded stack: pushing beyond max_ops drops the OLDEST entries and
    # keeps the undo chain consistent
    cap = 8
    hist = EditHistory(max_ops=cap)
    state = SplatState(50)

    def record(i):
        before = state.flags.copy()
        changed = state.select_indices([i], 'add')
        hist.push({'label': f'op{i}',
                   'indices': changed,
                   'before': before[changed].copy(),
                   'after': state.flags[changed].copy()})

    n_push = cap + 5
    for i in range(n_push):
        record(i)
    assert len(hist.ops) == cap, 'history grew past max_ops'
    assert hist.cursor == cap
    # oldest ops dropped, newest kept
    assert hist.ops[0]['label'] == f'op{n_push - cap}'
    assert hist.ops[-1]['label'] == f'op{n_push - 1}'

    # the full remaining chain undoes cleanly: the first (cap) selections
    # from the dropped ops survive, the rest are unwound
    while hist.can_undo:
        hist.undo(state)
    assert state.num_selected == n_push - cap
    for i in range(n_push - cap):
        assert state.flags[i] & State.SELECTED

    # and redoes back to the full selection
    while hist.can_redo:
        hist.redo(state)
    assert state.num_selected == n_push

    # default cap is 64
    assert EditHistory().max_ops == 64


def test_edit_history_transform_kind():
    # Phase 3: a 'transform' op carries geometry payloads EditHistory must NOT
    # apply itself — undo/redo return (direction, op) for the caller to handle,
    # and must leave the flag state untouched.
    hist = EditHistory()
    state = SplatState(20)
    state.select_indices([1, 2, 3], 'set')     # a live selection to protect
    flags_before = state.flags.copy()

    idx = np.array([4, 5])
    op = {
        'label': 'move',
        'kind': 'transform',
        'indices': idx,
        'before': {'positions': np.zeros((2, 3), np.float32)},
        'after': {'positions': np.ones((2, 3), np.float32)},
    }
    hist.push(op)

    # undo returns ('undo', op) and does NOT change the selection flags
    res = hist.undo(state)
    assert res == ('undo', op) or (res[0] == 'undo' and res[1] is op), res
    assert np.array_equal(state.flags, flags_before), 'transform undo touched flags'
    assert not hist.can_undo

    # redo returns ('redo', op), still no flag change
    res = hist.redo(state)
    assert res[0] == 'redo' and res[1] is op, res
    assert np.array_equal(state.flags, flags_before), 'transform redo touched flags'

    # a flags op still applies as before AND returns the tuple
    before = state.flags.copy()
    changed = state.select_indices([9], 'add')
    hist.push({'label': 'sel', 'indices': changed,
               'before': before[changed].copy(),
               'after': state.flags[changed].copy()})
    assert state.flags[9] & State.SELECTED
    res = hist.undo(state)
    assert res[0] == 'undo' and not (state.flags[9] & State.SELECTED), res

    # missing 'kind' defaults to flags (backward compatible)
    st2 = SplatState(5)
    h2 = EditHistory()
    b = st2.flags.copy()
    ch = st2.select_indices([0], 'set')
    h2.push({'label': 'legacy', 'indices': ch,
             'before': b[ch].copy(), 'after': st2.flags[ch].copy()})
    h2.undo(st2)
    assert st2.num_selected == 0


def test_serialize_rows():
    import base64
    import zlib

    # roundtrip a set of absolute source rows (order preserved)
    rows = np.array([0, 5, 17, 999, 1_000_000], np.int64)
    s = serialize_rows(rows)
    assert isinstance(s, str)
    back = deserialize_rows(s)
    assert back.dtype == np.int64
    assert np.array_equal(back, rows)

    # header is the index count as an 8-byte LE uint64
    payload = base64.b64decode(s)
    assert int.from_bytes(payload[:8], 'little') == rows.size
    zlib.decompress(payload[8:])   # valid zlib blob

    # empty roundtrips to an empty array; None too
    assert deserialize_rows(serialize_rows(np.zeros(0, np.int64))).size == 0
    assert deserialize_rows('').size == 0
    assert deserialize_rows(None).size == 0

    # unsorted / duplicate rows survive verbatim (take() relies on order)
    weird = np.array([9, 2, 2, 7, 0], np.int64)
    assert np.array_equal(deserialize_rows(serialize_rows(weird)), weird)

    # corrupt payloads raise ValueError (never garbage indices)
    good = (2).to_bytes(8, 'little')
    for bad in ('!!!not-base64!!!',
                base64.b64encode(b'\x01\x02').decode(),            # truncated
                base64.b64encode(good + b'junk').decode(),         # bad zlib
                # header count (2) disagrees with the single decoded int64
                base64.b64encode(good + zlib.compress(
                    np.array([1], np.int64).tobytes())).decode()):
        try:
            deserialize_rows(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'corrupt rows {bad!r} did not raise')


def main():
    test_selection()
    test_visibility_and_delete()
    test_serialize_header_and_mismatch()
    test_serialize_roundtrip()
    test_edit_history()
    test_edit_history_cap()
    test_edit_history_transform_kind()
    test_serialize_rows()
    print('all splat_state tests passed')


if __name__ == '__main__':
    main()
