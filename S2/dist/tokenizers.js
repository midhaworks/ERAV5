// Copyright 2026 Avnish Midha. All rights reserved.
// Author: Avnish Midha
// GitHub: avnishbm
// Purpose: Run the submitted PieceVocab and BPE tokenizers in the browser.

(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.ReviewTokenizers = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const ESCAPE = "\u0000UNICODE_ESCAPE";
  const HEX = Array.from({ length: 16 }, (_, value) => `\u0000HEX_${value.toString(16).toUpperCase()}`);

  function splitPieces(text) {
    const pieces = text.match(/\s*\S+/gu) || [];
    const consumed = pieces.reduce((total, piece) => total + piece.length, 0);
    if (consumed < text.length) pieces.push(text.slice(consumed));
    return pieces;
  }

  class PieceVocab {
    constructor(bundle) {
      this.vocab = bundle.vocab;
      this.tokenToId = new Map(this.vocab.map((token, id) => [token, id]));
      this.wholePieces = new Set(bundle.whole_pieces);
    }

    encode(text) {
      const ids = [];
      for (const piece of splitPieces(text)) {
        if (this.wholePieces.has(piece)) {
          ids.push(this.tokenToId.get(piece));
          continue;
        }
        for (const char of Array.from(piece)) {
          if (this.tokenToId.has(char)) ids.push(this.tokenToId.get(char));
          else {
            ids.push(this.tokenToId.get(ESCAPE));
            for (const digit of char.codePointAt(0).toString(16).toUpperCase().padStart(6, "0")) {
              ids.push(this.tokenToId.get(HEX[parseInt(digit, 16)]));
            }
          }
        }
      }
      return ids;
    }

    decode(ids) {
      let output = "";
      for (let index = 0; index < ids.length; index += 1) {
        const token = this.vocab[ids[index]];
        if (token === ESCAPE) {
          const digits = ids.slice(index + 1, index + 7).map(id => {
            const value = HEX.indexOf(this.vocab[id]);
            if (value < 0) throw new Error("Invalid Unicode escape");
            return value.toString(16);
          }).join("");
          if (digits.length !== 6) throw new Error("Truncated Unicode escape");
          output += String.fromCodePoint(parseInt(digits, 16));
          index += 6;
        } else if (HEX.includes(token)) {
          throw new Error("Unexpected hexadecimal token");
        } else output += token;
      }
      return output;
    }

    token(id) { return this.vocab[id]; }
  }

  function metaspaceSegments(text) {
    const converted = text.replaceAll(" ", "▁");
    if (!converted) return [];
    const segments = [];
    let start = 0;
    for (let index = 1; index < converted.length; index += 1) {
      if (converted[index] === "▁") {
        segments.push(converted.slice(start, index));
        start = index;
      }
    }
    segments.push(converted.slice(start));
    return segments.filter(Boolean);
  }

  class BPE {
    constructor(bundle) {
      this.vocabMap = bundle.model.vocab;
      this.vocab = [];
      for (const [token, id] of Object.entries(this.vocabMap)) this.vocab[id] = token;
      this.unk = bundle.model.unk_token;
      this.unkId = this.vocabMap[this.unk];
      this.mergeRanks = new Map(bundle.model.merges.map((pair, rank) => [`${pair[0]}\u0000${pair[1]}`, rank]));
    }

    encode(text) {
      const ids = [];
      for (const segment of metaspaceSegments(text)) {
        let symbols = Array.from(segment).map(char => this.vocabMap[char] === undefined ? this.unk : char);
        while (symbols.length > 1) {
          let bestRank = Infinity;
          let bestIndex = -1;
          for (let index = 0; index < symbols.length - 1; index += 1) {
            const rank = this.mergeRanks.get(`${symbols[index]}\u0000${symbols[index + 1]}`);
            if (rank !== undefined && rank < bestRank) {
              bestRank = rank;
              bestIndex = index;
            }
          }
          if (bestIndex < 0) break;
          symbols.splice(bestIndex, 2, symbols[bestIndex] + symbols[bestIndex + 1]);
        }
        for (const symbol of symbols) ids.push(this.vocabMap[symbol] ?? this.unkId);
      }
      return ids;
    }

    decode(ids) {
      return ids.map(id => this.vocab[id]).filter(token => token !== this.unk).join("").replaceAll("▁", " ");
    }

    token(id) { return this.vocab[id]; }
  }

  function faithfulUnits(text) {
    return (text.match(/[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]/gu) || []).length;
  }

  return { PieceVocab, BPE, faithfulUnits, splitPieces, metaspaceSegments };
});
