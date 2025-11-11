{-# LANGUAGE BangPatterns #-}
{-# OPTIONS_GHC -Wall #-}

-- =====================================================================
-- Project 2: Musician
-- Author: Quynh Anh Ha
-- 
-- Purpose: Implementation of both the performer and composer roles in
--          the Musician game, using an optimal guessing strategy.
-- =====================================================================

-- Overview:
-- This module implements a complete solution for the Musician guessing game,
-- a two-player logical deduction game. One player (the composer) selects a 
-- three-pitch musical chord as a target. The other player (the performer) 
-- attempts to guess it in as few guesses as possible, receiving feedback 
-- after each guess about how close they are.
--
-- The Game:
-- Each pitch consists of a musical note (A-G) and an octave (1-3), giving 
-- 21 possible pitches. A valid chord contains exactly 3 distinct pitches, 
-- yielding 1330 possible targets. After each guess, the composer provides:
--   * Number of pitches that exactly match the target (correct pitch)
--   * Number of pitches with the right note but wrong octave (correct note)
--   * Number of pitches with the right octave but wrong note (correct octave)
--
-- Pitches are only counted once: exact matches are not also counted as 
-- correct notes or octaves.
--
-- Strategy:
--
-- 1. Candidate Pruning: Maintains a set of possible targets, eliminating
--    any that are inconsistent with previous feedback.
--
-- 2. Hybrid Scoring: Evaluates each potential guess using three metrics:
--      * Expected remaining candidates (minimize average case)
--      * Worst-case remaining candidates (minimize worst case)
--      * Information entropy (maximize information gain)
--
-- 3. Two-Ply Lookahead: When the candidate set is small (≤50), performs
--    limited lookahead by simulating the next guess for each possible answer,
--    choosing the guess that minimizes expected remaining candidates after
--    the subsequent move.
-- =====================================================================

module Proj2 (Pitch, toPitch, feedback, GameState, initialGuess, nextGuess) where

-- ---------------------------------------------------------------------
--  Imports 
-- ---------------------------------------------------------------------

import Data.List (sortOn, minimumBy, foldl', (\\), intersect)
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)

-- =====================================================================
--  PART I: TYPE DEFINITIONS
-- =====================================================================

-- ---------------------------------------------------------------------
--  Core Musical Types
-- ---------------------------------------------------------------------

-- | Musical note (A..G).
data Note = A | B | C | D | E | F | G
  deriving (Eq, Ord, Enum, Bounded)

-- | Octave (only 1..3 are valid).
newtype Octave = Oct Int
  deriving (Eq, Ord)

-- | A pitch is a (Note, Octave) pair.
data Pitch = P !Note !Octave
  deriving (Eq, Ord)

-- | Show a Pitch exactly as required, e.g., "A1", "G3".
instance Show Pitch where
  show (P n (Oct o)) = noteChar n : show o

-- | A chord is exactly three distinct pitches in ascending order.
type Chord = [Pitch]

-- | Feedback answer: (correct pitches, correct notes, correct octaves)
type Answer = (Int,Int,Int)

-- ---------------------------------------------------------------------
--  Game State Type
-- ---------------------------------------------------------------------

-- | Game state carries the set of remaining candidate targets.
--   Strict field prevents storing thunks.
data GameState = GS {candidates :: ![Chord]}

-- ---------------------------------------------------------------------
--  Helper Constants
-- ---------------------------------------------------------------------

-- | All possible notes
allNotes :: [Note]
allNotes = [minBound .. maxBound]

-- | All valid octaves
allOctaves :: [Octave]
allOctaves = [Oct 1, Oct 2, Oct 3]

-- | All possible pitches (21 total)
allPitches :: [Pitch]
allPitches = [P n o | n <- allNotes, o <- allOctaves]

-- =====================================================================
--  PART II: REQUIRED API FUNCTIONS
-- =====================================================================

-- ---------------------------------------------------------------------
--  Pitch Parsing and Printing
-- ---------------------------------------------------------------------

-- | Parse a pitch name like \"A1\"..\"G3\".
--   Accepts exactly two characters: an uppercase note A..G and a digit 1..3.
--   Returns Nothing for anything else (lowercase, spaces, A10, etc.).
toPitch :: String -> Maybe Pitch
toPitch [cNote, cOct] = do
  n <- charToNote cNote
  o <- charToOct  cOct
  pure (P n (Oct o))
toPitch _ = Nothing

-- | Convert a note to its uppercase character representation.
--   Used by the Show instance for Pitch.
noteChar :: Note -> Char
noteChar A = 'A'
noteChar B = 'B'
noteChar C = 'C'
noteChar D = 'D'
noteChar E = 'E'
noteChar F = 'F'
noteChar G = 'G'

-- | Parse a single character into a Note.
--   Only accepts uppercase letters A through G.
--   
--   Returns Nothing for any other character (including lowercase).
charToNote :: Char -> Maybe Note
charToNote 'A' = Just A
charToNote 'B' = Just B
charToNote 'C' = Just C
charToNote 'D' = Just D
charToNote 'E' = Just E
charToNote 'F' = Just F
charToNote 'G' = Just G
charToNote _   = Nothing

-- | Parse a digit character into an octave number.
--   Only accepts the digits '1', '2', or '3'.
--   
--   Returns Nothing for any other character.
charToOct :: Char -> Maybe Int
charToOct '1' = Just 1
charToOct '2' = Just 2
charToOct '3' = Just 3
charToOct _   = Nothing

-- ---------------------------------------------------------------------
--  Feedback Function
--  Exact → Notes-only → Octaves-only (no double counting)
-- ---------------------------------------------------------------------

-- | Compute (correct pitches, correct notes, correct octaves).
--   1) Exact matches (pitch+octave) are counted first and removed.
--   2) Notes-only matches are counted on the remainder (min of note counts).
--   3) Octaves-only matches are then counted on the remainder (min of octave counts).
feedback :: [Pitch] -> [Pitch] -> (Int, Int, Int)
feedback target guess =
  let exacts = target `intersect` guess
      p      = length exacts
      -- remove exact matches from both sides (one-for-one)
      tRem   = target \\ exacts
      gRem   = guess  \\ exacts

      -- counts per note/octave on the remainder
      tNoteCnt = noteCounts tRem
      gNoteCnt = noteCounts gRem
      nOnly    = sum (zipWith min tNoteCnt gNoteCnt)

      -- after accounting for notes-only, we conceptually consume those.
      -- but since each pitch contributes exactly one note and one octave,
      -- and we've removed exacts already, counting octaves on the *same remainder*
      -- yields the correct tally for "octave-only" under the spec.
      tOctCnt = octCounts tRem
      gOctCnt = octCounts gRem
      oOnly   = sum (zipWith min tOctCnt gOctCnt)
  in (p, nOnly, oOnly)

-- | Count how many of each note appear in a chord (A..G)
noteCounts :: [Pitch] -> [Int]
noteCounts = go 0 0 0 0 0 0 0
  where
    go !a !b !c !d !e !f !g [] = [a,b,c,d,e,f,g]
    go !a !b !c !d !e !f !g (P n _ : xs) =
      case n of
        A -> go (a+1) b     c     d     e     f     g     xs
        B -> go a     (b+1) c     d     e     f     g     xs
        C -> go a     b     (c+1) d     e     f     g     xs
        D -> go a     b     c     (d+1) e     f     g     xs
        E -> go a     b     c     d     (e+1) f     g     xs
        F -> go a     b     c     d     e     (f+1) g     xs
        G -> go a     b     c     d     e     f     (g+1) xs

-- | Count how many of each octave appear in a chord (1..3)
octCounts :: [Pitch] -> [Int]
octCounts = go 0 0 0
  where
    go !o1 !o2 !o3 [] = [o1,o2,o3]
    go !o1 !o2 !o3 (P _ (Oct o) : xs) =
      case o of
        1 -> go (o1+1) o2     o3     xs
        2 -> go o1     (o2+1) o3     xs
        _ -> go o1     o2     (o3+1) xs

-- =====================================================================
--  PART III: GAME INFRASTRUCTURE
-- =====================================================================

-- ---------------------------------------------------------------------
--  Candidate Generation (21 pitches → 1330 chords)
-- ---------------------------------------------------------------------

-- | All legal chords: choose 3 distinct pitches, order irrelevant.
--   The generator preserves ascending order, so each chord is canonical.
allChords :: [[Pitch]]
allChords = combinations 3 allPitches

-- | Generate all k-element combinations from a list without replacement.
--   The order of elements in the input is preserved in the output.
--   
--   This produces all possible ways to choose k items from the input list,
--   maintaining the relative order. Used to generate all valid 3-pitch chords.
combinations :: Int -> [a] -> [[a]]
combinations 0 _      = [[]]         -- Base case: 0 items = empty combination
combinations _ []     = []           -- No items left to choose from
combinations k (y:ys)
  | k < 0             = []           -- Invalid: can't choose negative items
  | otherwise         = map (y:) (combinations (k-1) ys) ++ combinations k ys
                        -- Include y in combinations OR skip y

-- ---------------------------------------------------------------------
--  Initial Guess
-- ---------------------------------------------------------------------

-- | First guess used to start the game.
--   Chosen to maximize coverage across the pitch space: low note in octave 1,
--   middle note in octave 2, high note in octave 3. This spread provides good
--   initial information about the target regardless of its composition.
firstGuess :: [Pitch]
firstGuess = [P A (Oct 1), P D (Oct 2), P G (Oct 3)]

-- | First move: return the opener and the fresh candidate set.
--   We keep the state simple here; pruning happens after the first feedback.
initialGuess :: ([Pitch], GameState)
initialGuess = (firstGuess, GS allChords)

-- =====================================================================
--  PART IV: GUESSING STRATEGY
-- =====================================================================

-- ---------------------------------------------------------------------
--  Configuration Parameters
-- ---------------------------------------------------------------------

-- | Only apply two-ply lookahead when candidate set ≤ this size.
--   Two-ply lookahead is expensive (requires simulating all possible next moves),
--   but becomes more valuable when the search space is smaller. This threshold
--   balances computation time against quality improvement.
twoPlyThreshold :: Int
twoPlyThreshold = 50

-- | Examine this many top one-ply guesses when doing two-ply.
--   Rather than evaluating two-ply for all candidates (expensive), we only
--   examine the top K candidates from one-ply scoring, then do full two-ply
--   evaluation on just those.
twoPlyTopK :: Int
twoPlyTopK = 4

-- ---------------------------------------------------------------------
--  Consistency Pruning
-- ---------------------------------------------------------------------

-- | Remove any candidate chords that are inconsistent with the
--   feedback received for the previous guess.
--   A candidate t survives if feedback t guess == actualFeedback.
prune :: Chord               -- ^ previous guess
      -> (Int,Int,Int)       -- ^ feedback for that guess
      -> [Chord]             -- ^ current candidate set
      -> [Chord]
prune guess fb candidates =
  [ t | t <- candidates, feedback t guess == fb ]

-- ---------------------------------------------------------------------
--  Scoring Framework
--  We score a candidate guess g against the current candidate set S by
--  building a histogram over answers a = feedback t g for t ∈ S.
--  From that we compute:
--    E  = (Σ H[a]^2) / N           -- expected remaining size
--    W  = max_a H[a]               -- worst-case remaining size
--    Ent= - Σ p[a] * log p[a]      -- answer entropy (higher is better)
--  We then pick the g that minimizes (E, W, -Ent).
-- ---------------------------------------------------------------------

-- | Score record for a candidate guess
data Score = Score
  { scExp   :: !Double  -- ^ Expected remaining candidates
  , scWorst :: !Int     -- ^ Worst-case remaining candidates
  , scEnt   :: !Double  -- ^ Information entropy
  } deriving (Show)

-- | Lexicographic compare on (E asc, W asc, Ent desc).
compareScore :: Score -> Score -> Ordering
compareScore (Score e1 w1 h1) (Score e2 w2 h2) =
  compare e1 e2 <> compare w1 w2 <> compare h2 h1  -- note h2 vs h1 to prefer larger entropy

-- | Histogram of answers produced by guessing g against S.
histogram :: [Chord] -> Chord -> Map Answer Int
histogram s g =
  foldl' (\m t ->
            let !a = feedback t g
            in  Map.insertWith (+) a 1 m)
         Map.empty
         s

-- | Compute the hybrid score for guess g on candidate set s.
scoreGuess :: [Chord] -> Chord -> Score
scoreGuess s g =
  let n     = length s
      nD    = fromIntegral n :: Double
      hist  = histogram s g          
      cnts  = Map.elems hist
      eExp  = (sum [ fromIntegral c * fromIntegral c | c <- cnts ]) / nD
      wMax  = if null cnts then 0 else maximum cnts
      probs = [ fromIntegral c / nD | c <- cnts, c > 0 ]
      ent   = negate (sum [ p * log p | p <- probs ])
  in  Score eExp wMax ent

-- | Extract a sortable key from a candidate guess for ranking.
--   Returns a 3-tuple (E, W, -Ent) where lower values are better.
--   
--   The tuple is compared lexicographically by Haskell's default Ord instance:
--   first by expected value E (lower better), then by worst-case W (lower better),
--   finally by negative entropy -Ent (lower better, i.e., higher entropy better).
--   
--   This allows efficient sorting and selection of top candidates.
scoreKey :: [Chord] -> Chord -> (Double, Int, Double)
scoreKey s g =
  let sc = scoreGuess s g
  in  (scExp sc, scWorst sc, negate (scEnt sc))

-- ---------------------------------------------------------------------
--  One-Ply Strategy
-- ---------------------------------------------------------------------

-- | Choose the best next guess from the current candidate set (1-ply).
--   Deterministic tie-break via compareScore.
chooseGuess1ply :: [Chord] -> Chord
chooseGuess1ply [] = error "chooseGuess1ply: empty candidate set"
chooseGuess1ply s  = minimumBy cmp s
  where
    cmp g1 g2 = compareScore (scoreGuess s g1) (scoreGuess s g2)

-- ---------------------------------------------------------------------
--  Two-Ply Lookahead Strategy
-- ---------------------------------------------------------------------

-- | Take the top-k one-ply guesses by our hybrid score.
--   Sorts all candidates by their scoreKey (lower is better) and returns
--   the first k candidates. Used to reduce the search space for two-ply
--   evaluation by pre-filtering to the most promising candidates.
--   
--   Arguments:
--   
--   * k - Number of top candidates to return
--   * s - Current candidate set
--   
--   Returns: The k best candidates according to one-ply scoring
topKByScore :: Int -> [Chord] -> [Chord]
topKByScore k s =
  take k (sortOn (scoreKey s) s)

-- | Two-ply expected remaining size for a candidate g:
--   For each answer bucket a from guessing g, we:
--     * restrict to S' = { t in S | feedback t g == a }
--     * pick best one-ply response g' in S' (no recursion!)
--     * use scExp score(S', g') as the next expected remaining
--   Then weight by bucket probability and sum.
twoPlyScore :: [Chord] -> Chord -> Double
twoPlyScore s g =
  let n  = length s
      nD = fromIntegral n :: Double
      hist = histogram s g
      bucket ans cnt =
        let s'  = [ t | t <- s, feedback t g == ans ]
            g'  = chooseGuess1ply s'
            sc' = scoreGuess s' g'
        in  (fromIntegral cnt / nD) * scExp sc'
  in  Map.foldrWithKey (\ans cnt acc -> bucket ans cnt + acc) 0 hist

-- | Final chooser: use two-ply when the set is small, otherwise stick to one-ply.
chooseGuess :: [Chord] -> Chord
chooseGuess s =
  let !len = length s
  in
  if len <= twoPlyThreshold
     then
       let pool = topKByScore twoPlyTopK s
       in  minimumBy (\g1 g2 -> compare (twoPlyScore s g1) (twoPlyScore s g2)) pool
     else
       chooseGuess1ply s

-- =====================================================================
--  PART V: MAIN GAME LOOP
-- =====================================================================

-- | Given (previous guess, state) and feedback triple,
--   produce (next guess, updated state).
--   Steps:
--     1) Prune the candidate list with the new feedback.
--     2) If one candidate remains → guess it (game won).
--     3) Otherwise → choose next guess via chooseGuess / two-ply gate.
nextGuess :: ([Pitch], GameState) -> (Int,Int,Int) -> ([Pitch], GameState)
nextGuess (prevGuess, stPrev) fb =
  let sPrev   = candidates stPrev
      sPruned = prune prevGuess fb sPrev
      next =
        case sPruned of
          []   -> error "No candidates left – inconsistent feedback."
          [x]  -> x
          _    -> chooseGuess sPruned
  in  (next, GS sPruned)
