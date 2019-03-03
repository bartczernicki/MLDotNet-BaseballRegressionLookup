﻿using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace MLDotNet_BaseballRegressionLookup
{
    public class MLBBaseballBatter
    {
        [LoadColumn(0), ColumnName("InductedToHallOfFame")]
        public bool InductedToHallOfFame { get; set; }

        [LoadColumn(1), ColumnName("OnHallOfFameBallot")]
        public bool OnHallOfFameBallot { get; set; }

        [LoadColumn(2), ColumnName("FullPlayerName")]
        public string FullPlayerName { get; set; }

        [LoadColumn(3), ColumnName("YearsPlayed")]
        public float YearsPlayed { get; set; }

        [LoadColumn(4), ColumnName("AB")]
        public float AB { get; set; }

        [LoadColumn(5), ColumnName("R")]
        public float R { get; set; }

        [LoadColumn(6), ColumnName("H")]
        public float H { get; set; }

        [LoadColumn(7), ColumnName("Doubles")]
        public float Doubles { get; set; }

        [LoadColumn(8), ColumnName("Triples")]
        public float Triples { get; set; }

        [LoadColumn(9), ColumnName("HR")]
        public float HR { get; set; }

        [LoadColumn(10), ColumnName("RBI")]
        public float RBI { get; set; }

        [LoadColumn(11), ColumnName("SB")]
        public float SB { get; set; }

        [LoadColumn(12), ColumnName("BattingAverage")]
        public float BattingAverage { get; set; }

        [LoadColumn(13), ColumnName("SluggingPct")]
        public float SluggingPct { get; set; }

        [LoadColumn(14), ColumnName("AllStarAppearances")]
        public float AllStarAppearances { get; set; }

        [LoadColumn(15), ColumnName("MVPs")]
        public float MVPs { get; set; }

        [LoadColumn(16), ColumnName("TripleCrowns")]
        public float TripleCrowns { get; set; }

        [LoadColumn(17), ColumnName("GoldGloves")]
        public float GoldGloves { get; set; }

        [LoadColumn(18), ColumnName("MajorLeaguePlayerOfTheYearAwards")]
        public float MajorLeaguePlayerOfTheYearAwards { get; set; }

        [LoadColumn(19), ColumnName("TB")]
        public float TB { get; set; }

        [LoadColumn(20), ColumnName("LastYearPlayed")]
        public float LastYearPlayed { get; set; }

        [LoadColumn(21), ColumnName("ID")]
        public float ID { get; set; }
    }

    public class HitsPredictions
    {
            [ColumnName("Score")]
            public float Hits;
    }
}
