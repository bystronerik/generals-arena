//! How far a thing is, in moves.
//!
//! **Range is measured in moves**, not Manhattan steps: a bounded BFS out of
//! a general over every cell not remembered as a mountain. That stays a
//! superset — army travels one orthogonal step per turn and mountains are the
//! only permanently impassable cells (RULES.md §01), so nothing that could
//! arrive within the budget is ever excluded — and it is tighter than raw
//! Manhattan distance, which counts armies that a wall stands between.
//!
//! One `Reach` is shared by the whole layer and recomputed per question, so
//! the per-move path never grows the heap.

use crate::board::memory::Memory;
use crate::board::obs::DIRECTIONS;
use crate::io::wire::Observation;

/// Distance for a cell the BFS never reached: walled off, or further than the
/// step budget it was given.
const UNREACHED: i32 = i32::MAX;

/// A bounded breadth-first reach out of one cell, plus its scratch. Held
/// across turns so the per-move path never grows the heap.
#[derive(Debug, Default)]
pub struct Reach {
    dist: Vec<i32>,
    queue: Vec<usize>,
}

impl Reach {
    /// Moves from `origin` to every cell, over everything not remembered as a
    /// mountain, capped at `max_steps`. Overwrites whatever the last call
    /// computed.
    ///
    /// Only *remembered* mountains block. Fog does not: a cell we cannot see
    /// may well be walkable, and pretending otherwise would drop real
    /// threats.
    pub fn compute(&mut self, obs: &Observation, mem: &Memory, origin: usize, max_steps: i32) {
        let (h, w) = (obs.h, obs.w);
        self.dist.clear();
        self.dist.resize(h * w, UNREACHED);
        self.queue.clear();
        self.dist[origin] = 0;
        self.queue.push(origin);

        let mut head = 0;
        while head < self.queue.len() {
            let cell = self.queue[head];
            head += 1;
            let steps = self.dist[cell];
            if steps == max_steps {
                continue;
            }
            let (row, col) = ((cell / w) as i32, (cell % w) as i32);
            for (dr, dc) in DIRECTIONS {
                let (nr, nc) = (row + dr, col + dc);
                if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 {
                    continue;
                }
                let next = nr as usize * w + nc as usize;
                if mem.mountains[next] || self.dist[next] != UNREACHED {
                    continue;
                }
                self.dist[next] = steps + 1;
                self.queue.push(next);
            }
        }
    }

    /// Is this cell inside the budget the last `compute` was given?
    pub fn reached(&self, cell: usize) -> bool {
        self.dist[cell] != UNREACHED
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::wire::TYPE_MOUNTAIN;
    use crate::tactics::common::fixtures::{at, frame, put, H, W};

    fn reach_from(obs: &Observation, origin: usize, max_steps: i32) -> Reach {
        let mut mem = Memory::new(H, W);
        mem.update(obs);
        let mut reach = Reach::default();
        reach.compute(obs, &mem, origin, max_steps);
        reach
    }

    #[test]
    fn reach_stops_at_the_step_budget() {
        let obs = frame(0);
        let reach = reach_from(&obs, at(4, 4), 2);
        assert!(reach.reached(at(4, 4))); // the origin itself
        assert!(reach.reached(at(4, 6))); // two moves
        assert!(!reach.reached(at(4, 7))); // three
        assert!(reach.reached(at(3, 5))); // two, around a corner
        assert!(!reach.reached(at(3, 6)));
    }

    #[test]
    fn reach_goes_around_remembered_mountains_and_never_through() {
        let mut obs = frame(0);
        for (row, col) in [(3, 5), (4, 5), (5, 5)] {
            put(&mut obs, row, col, TYPE_MOUNTAIN, 0, 0);
        }
        let reach = reach_from(&obs, at(4, 4), 3);
        assert!(!reach.reached(at(4, 5))); // the wall
        assert!(!reach.reached(at(4, 6))); // straight through it
        assert!(reach.reached(at(2, 5))); // over the top: 2 up, 1 right
    }

    #[test]
    fn reach_seals_a_cell_with_no_open_neighbour() {
        // (0, 0) has two neighbours and both are walls.
        let mut obs = frame(0);
        put(&mut obs, 0, 1, TYPE_MOUNTAIN, 0, 0);
        put(&mut obs, 1, 0, TYPE_MOUNTAIN, 0, 0);
        let reach = reach_from(&obs, at(1, 1), 8);
        assert!(!reach.reached(at(0, 0)));
    }
}
