import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

public class CleanSealedBiotSuite {
  static final String OUTPUT_DIRECTORY = "./outputs/";
  private static void createBoxInside(Model model, String name, double xmin, double xmax, double ymin, double ymax) {
    model.component("comp1").selection().create(name, "Box");
    model.component("comp1").selection(name).geom("geom1", 1);
    model.component("comp1").selection(name).set("entitydim", 1);
    model.component("comp1").selection(name).set("xmin", xmin);
    model.component("comp1").selection(name).set("xmax", xmax);
    model.component("comp1").selection(name).set("ymin", ymin);
    model.component("comp1").selection(name).set("ymax", ymax);
    model.component("comp1").selection(name).set("condition", "inside");
  }

  public static Model runOne(String tag, String outputCsv, double period, double cond, double burial,
      double xi1, double xi2, double xi3, double xi4, double kwave, boolean het, String tier, boolean keep) {
    long t0 = System.nanoTime();
    Model model = ModelUtil.create(tag);
    model.modelPath(OUTPUT_DIRECTORY);
    model.param().set("W", "2[m]");
    model.param().set("Hs", "0.58[m]");
    model.param().set("D", "0.10[m]");
    model.param().set("R", "D/2");
    model.param().set("T", Double.toString(period) + "[s]");
    model.param().set("freq0", "1/T");
    model.param().set("omega0", "2*pi*freq0");
    model.param().set("Kmean", Double.toString(cond) + "[m/s]");
    model.param().set("burial", Double.toString(burial));
    model.param().set("yc", "Hs-burial*D");
    model.param().set("rfcoef1", Double.toString(xi1));
    model.param().set("rfcoef2", Double.toString(xi2));
    model.param().set("rfcoef3", Double.toString(xi3));
    model.param().set("rfcoef4", Double.toString(xi4));
    model.param().set("rho_w", "1000[kg/m^3]");
    model.param().set("g0", "9.81[m/s^2]");
    model.param().set("phi", "0.396");
    model.param().set("Sw", "0.999");
    model.param().set("Kw", "1.95e9[Pa]");
    model.param().set("G", "1e7[Pa]");
    model.param().set("nu0", "0.32");
    model.param().set("E0", "2*G*(1+nu0)");
    model.param().set("storage", "phi*(1/Kw+(1-Sw)/(rho_w*g0*0.4[m]))");
    model.param().set("k_wave", Double.toString(kwave) + "[1/m]");
    model.param().set("P0", "1[Pa]");
    model.param().set("sig_rf", het ? "sqrt(log(1+0.35^2))" : "0");

    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 2);
    model.component("comp1").geom("geom1").lengthUnit("m");
    model.component("comp1").geom("geom1").create("rect", "Rectangle");
    model.component("comp1").geom("geom1").feature("rect").set("size", new String[] {"W", "Hs"});
    model.component("comp1").geom("geom1").feature("rect").set("pos", new String[] {"-W/2", "0"});
    model.component("comp1").geom("geom1").create("pipe", "Circle");
    model.component("comp1").geom("geom1").feature("pipe").set("r", "R");
    model.component("comp1").geom("geom1").feature("pipe").set("pos", new String[] {"0", "yc"});
    model.component("comp1").geom("geom1").create("dif", "Difference");
    model.component("comp1").geom("geom1").feature("dif").selection("input").set("rect");
    model.component("comp1").geom("geom1").feature("dif").selection("input2").set("pipe");
    model.component("comp1").geom("geom1").run();

    createBoxInside(model, "top", -1.01, 1.01, 0.5799, 0.5801);
    createBoxInside(model, "bottom", -1.01, 1.01, -1e-4, 1e-4);
    createBoxInside(model, "left", -1.0001, -0.9999, -1e-4, 0.5801);
    createBoxInside(model, "right", 0.9999, 1.0001, -1e-4, 0.5801);
    model.component("comp1").selection().create("pipeb", "Ball");
    model.component("comp1").selection("pipeb").geom("geom1", 1);
    model.component("comp1").selection("pipeb").set("entitydim", 1);
    model.component("comp1").selection("pipeb").set("posx", 0);
    model.component("comp1").selection("pipeb").set("posy", "yc");
    model.component("comp1").selection("pipeb").set("r", "1.01*R");
    model.component("comp1").selection("pipeb").set("condition", "inside");

    model.component("comp1").selection().create("fixedb", "Union");
    model.component("comp1").selection("fixedb").geom("geom1", 1);
    model.component("comp1").selection("fixedb").set("entitydim", 1);
    model.component("comp1").selection("fixedb").set("input", new String[] {"bottom", "left", "right", "pipeb"});
    model.component("comp1").selection().create("pressureb", "Union");
    model.component("comp1").selection("pressureb").geom("geom1", 1);
    model.component("comp1").selection("pressureb").set("entitydim", 1);
    model.component("comp1").selection("pressureb").set("input", new String[] {"top", "left", "right"});

    model.component("comp1").variable().create("var1");
    model.component("comp1").variable("var1").set("m1", "cos(0.5*pi*x/(W/2))*cos(pi*y/Hs)");
    model.component("comp1").variable("var1").set("m2", "sin(pi*x/(W/2))*cos(0.5*pi*y/Hs)");
    model.component("comp1").variable("var1").set("m3", "cos(1.5*pi*x/(W/2))*sin(pi*y/Hs)");
    model.component("comp1").variable("var1").set("m4", "sin(2*pi*x/(W/2))*sin(1.5*pi*y/Hs)");
    model.component("comp1").variable("var1").set("gfield", "(rfcoef1*m1+rfcoef2*m2+rfcoef3*m3+rfcoef4*m4)/2");
    model.component("comp1").variable("var1").set("vfield", "(m1^2+m2^2+m3^2+m4^2)/4");
    model.component("comp1").variable("var1").set("Kfield", "Kmean*exp(sig_rf*gfield-0.5*sig_rf^2*vfield)");

    model.material().create("fluid", "Common", "");
    model.material("fluid").propertyGroup("def").set("density", new String[] {"rho_w"});
    model.material("fluid").propertyGroup("def").set("compressibility", new String[] {"1/Kw"});

    model.component("comp1").material().create("soil", "PorousMedia");
    model.component("comp1").material("soil").selection().all();
    model.component("comp1").material("soil").set("porosity", "phi");
    model.component("comp1").material("soil").feature().create("fluid1", "Fluid", "comp1");
    model.component("comp1").material("soil").propertyGroup("def").set("density", new String[] {"2000[kg/m^3]"});
    model.component("comp1").material("soil").propertyGroup().create("Enu", "Enu", "Young's_modulus_and_Poisson's_ratio");
    model.component("comp1").material("soil").propertyGroup("Enu").set("E", new String[] {"E0"});
    model.component("comp1").material("soil").propertyGroup("Enu").set("nu", new String[] {"nu0"});

    model.component("comp1").physics().create("dl", "PorousMediaFlowDarcy", "geom1");
    model.component("comp1").physics("dl").feature("porous1").set("storageModelType", "userdef");
    model.component("comp1").physics("dl").feature("porous1").set("Sp", "storage");
    model.component("comp1").physics("dl").feature("porous1").feature("fluid1").set("fluidType", "compressibleLinearized");
    model.component("comp1").physics("dl").feature("porous1").feature("pm1").set("permeabilityModelType", "conductivity");
    model.component("comp1").physics("dl").feature("porous1").feature("pm1").set("K", new String[] {"Kfield","0","0","0","Kfield","0","0","0","Kfield"});
    model.component("comp1").physics("dl").create("p1", "Pressure", 1);
    model.component("comp1").physics("dl").feature("p1").selection().named("pressureb");
    model.component("comp1").physics("dl").feature("p1").set("p0", "P0*exp(i*k_wave*x)");

    model.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
    model.component("comp1").physics("solid").create("fix1", "Fixed", 1);
    model.component("comp1").physics("solid").feature("fix1").selection().named("fixedb");

    model.component("comp1").multiphysics().create("poro", "PoroelasticCoupling", 2);
    model.component("comp1").multiphysics("poro").selection().all();
    model.component("comp1").multiphysics("poro").set("Solid_physics", "solid");
    model.component("comp1").multiphysics("poro").set("PorousMediaFlow_physics", "dl");
    model.component("comp1").multiphysics("poro").set("alphaB_mat", "userdef");
    model.component("comp1").multiphysics("poro").set("alphaB", 1);

    long tm0 = System.nanoTime();
    model.component("comp1").mesh().create("mesh1");
    int hauto = 2; String hmax = "0.002[m]"; String hmin = "0.0005[m]";
    if ("coarse".equals(tier)) { hauto = 7; hmax = "0.020[m]"; hmin = "0.008[m]"; }
    else if ("medium".equals(tier)) { hauto = 4; hmax = "0.006[m]"; hmin = "0.002[m]"; }
    else if ("ultrafine".equals(tier)) { hauto = 1; hmax = "0.0008[m]"; hmin = "0.0002[m]"; }
    model.component("comp1").mesh("mesh1").feature("size").set("hauto", hauto);
    model.component("comp1").mesh("mesh1").create("sizePipe", "Size");
    model.component("comp1").mesh("mesh1").feature("sizePipe").selection().named("pipeb");
    model.component("comp1").mesh("mesh1").feature("sizePipe").set("custom", true);
    model.component("comp1").mesh("mesh1").feature("sizePipe").set("hmax", hmax);
    model.component("comp1").mesh("mesh1").feature("sizePipe").set("hmin", hmin);
    model.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");
    model.component("comp1").mesh("mesh1").run();
    int ne = model.component("comp1").mesh("mesh1").getNumElem();
    int nv = model.component("comp1").mesh("mesh1").getNumVertex();
    long tm1 = System.nanoTime();

    model.study().create("std1");
    model.study("std1").create("freq", "Frequency");
    model.study("std1").feature("freq").set("plist", "freq0");
    model.study("std1").createAutoSequences("all");
    long ts0 = System.nanoTime();
    model.sol("sol1").runAll();
    long ts1 = System.nanoTime();

    model.result().export().create("data1", "Data");
    model.result().export("data1").set("expr", new String[] {"x", "y", "real(p)/P0", "imag(p)/P0", "abs(p)/P0"});
    model.result().export("data1").set("filename", outputCsv);
    model.result().export("data1").set("separator", ",");
    model.result().export("data1").run();
    long t1 = System.nanoTime();

    System.out.println("TIMING_ROW," + tag + "," + tier + "," + het + "," + ne + "," + nv + ","
        + ((tm1 - tm0) / 1.0e6) + "," + ((ts1 - ts0) / 1.0e6) + "," + ((t1 - t0) / 1.0e6));
    if (!keep) {
      ModelUtil.remove(model.tag());
      return null;
    }
    return model;
  }

  public static Model run() {
    String base = OUTPUT_DIRECTORY;
    String[][] cases = {
      {"train_000", "train", "1.2", "1E-05", "0.75", "-0.32373543890692463", "-0.5150846423372833", "0.3095942679193938", "-0.00664273796788647", "3.2450310896560444", "false"},
      {"train_001", "train", "1.8", "8E-05", "1.5", "-0.757822681894033", "-1.4395656261382817", "-0.34965533766521906", "0.8779335485734611", "1.922011662342012", "false"},
      {"train_002", "train", "1.8", "8E-05", "0.75", "1.154904874838434", "1.1791763809052005", "0.10522732993356246", "0.14257783257634793", "1.922011662342012", "false"},
      {"train_003", "train", "1.2", "1E-05", "1.5", "1.3828362321864738", "0.16867184599773805", "0.6874593152462313", "-1.1163719273916657", "3.2450310896560444", "false"},
      {"train_004", "train", "1.8", "1E-05", "1.5", "-0.14461193212517787", "-0.05737429458881714", "0.5511747723151813", "1.307098910033273", "1.922011662342012", "false"},
      {"train_005", "train", "1.2", "8E-05", "0.75", "-0.947454570196192", "0.37224564059442905", "1.563445084190717", "1.0809942984252858", "3.2450310896560444", "false"},
      {"train_006", "train", "1.2", "8E-05", "1.5", "-0.5792748369547914", "0.3133805019342207", "-0.1462096100658848", "0.4429049876504405", "3.2450310896560444", "false"},
      {"train_007", "train", "1.8", "1E-05", "0.75", "-0.003777356647902234", "1.353107436426656", "-0.9868341960701996", "0.0766053266831696", "1.922011662342012", "false"},
      {"train_008", "train", "1.5546910973890373", "5.34160214517795E-05", "0.8949448081663635", "-0.9644921205498707", "-0.6794648749397363", "-0.030060770685285548", "-1.5392997799272332", "2.29602968973313", "false"},
      {"train_009", "train", "1.4015378128846445", "4.07451745544693E-05", "0.7816823301614189", "0.8940200105526029", "1.4562662624057534", "-0.9274274290558445", "0.23554205434534187", "2.6220344398519275", "false"},
      {"train_010", "train", "1.5491564061999916", "4.16406336447846E-05", "0.954076331225677", "1.012051344656166", "-0.607678519995029", "1.0954353301728101", "0.7914246279413969", "2.3062926540470854", "false"},
      {"train_011", "train", "1.4598584841254696", "5.10193044380457E-05", "1.051135736118686", "0.7783837776252157", "-0.4733678083297352", "0.013726401301047129", "0.6204975648064537", "2.4867123383433403", "false"},
      {"train_012", "train", "1.3498971007253673", "1.0471481472044E-05", "1.2285781334538197", "-0.6302177908088168", "-0.32664759270155785", "-0.47926509278658197", "-0.6782787116930605", "2.7559269591899516", "false"},
      {"train_013", "train", "1.5932614951011148", "6.86730331567074E-05", "1.330907852646023", "-1.0972276055765342", "1.056563264809918", "0.20102504762426782", "0.7058972820092208", "2.22714374259019", "false"},
      {"train_014", "train", "1.5147580788364063", "2.21626478110968E-05", "0.8617207099708398", "-0.13066584618130508", "0.2733743054702878", "-1.0811525125045132", "-1.050079503521875", "2.3723605893072857", "false"},
      {"train_015", "train", "1.6132881797010394", "5.91441437422296E-05", "1.4356956699339902", "0.5626162899830048", "-1.3219935765652286", "-0.6924660288290116", "0.3908392315495909", "2.1930949249346408", "false"},
      {"train_016", "train", "1.6880440410743622", "2.48231440895508E-05", "1.4698409365691654", "0.3370415168042853", "-0.17972208141182336", "-0.3616252099264696", "-0.648492345041244", "2.0752663960229607", "false"},
      {"train_017", "train", "1.2800951405354617", "1.1381720904934E-05", "0.9087359151604615", "0.20943200274952417", "0.0941112383332076", "1.2617973346040876", "-0.584012947883575", "2.962414971818834", "false"},
      {"train_018", "train", "1.5756570205327836", "3.21412100485012E-05", "0.8489347856787451", "-0.8144265054449913", "-0.23619977828290503", "0.7981732894587391", "0.06549241497210632", "2.258026596174928", "false"},
      {"train_019", "train", "1.6339929055656262", "1.52557686610177E-05", "1.3602600333147836", "-1.3258723062850357", "-0.13399816875819673", "0.6286319933124875", "-0.3211809228007796", "2.1590463609677992", "false"},
      {"train_020", "train", "1.5358125229675248", "7.5532474535802E-05", "1.109729813004586", "0.059113685585737985", "0.533477483285698", "0.42675249344052946", "-0.7799139200905395", "2.331446853816718", "false"},
      {"train_021", "train", "1.6782137879150076", "2.13765373431102E-05", "1.4547092794534155", "0.45736007370204146", "-0.8556852583215692", "-0.8147153816535807", "-0.4988359123532531", "2.089982768421361", "false"},
      {"train_022", "train", "1.7060799923115324", "1.31332881897554E-05", "1.1471028388870872", "0.3813593789408137", "-1.1136807709940009", "-1.3263242597930067", "0.5610173391583158", "2.048833553645448", "false"},
      {"train_023", "train", "1.7401534184581187", "3.06928334390149E-05", "1.32288566109685", "0.6626830072906014", "0.4382445284742763", "0.4589394142189664", "-1.2252642951133483", "2.0008079338823412", "false"},
      {"train_024", "train", "1.3218504094208385", "2.02914522930945E-05", "1.2538157842244044", "0.16369692123963483", "0.6108855721917571", "-1.3767435448551355", "-0.20418844747327158", "2.8350521547376353", "false"},
      {"train_025", "train", "1.4988481540495848", "1.41877542960177E-05", "0.7879425816307983", "-1.3825419378426242", "0.6764357790121732", "0.9735338371747745", "-0.42551793218958783", "2.4043190036432907", "false"},
      {"train_026", "train", "1.722884951843647", "2.76935256386994E-05", "1.3935367824365212", "1.29694127716063", "-0.8226044299637159", "-0.5897515232924613", "1.487748407851767", "2.0248434546043517", "false"},
      {"train_027", "train", "1.3784644486604358", "1.83599069032445E-05", "1.1846144923580373", "0.09058621829843576", "-0.9958297138449352", "-0.22581548242736035", "-0.08939575081555544", "2.680083120589916", "false"},
      {"train_028", "train", "1.476455617212329", "1.1959355858537E-05", "0.8111055558420008", "-0.21853086814366118", "0.8970846246122296", "0.2626405391973307", "-0.24723548896162212", "2.450905484638969", "false"},
      {"train_029", "train", "1.7713553876630521", "2.92220443571984E-05", "1.2643146909591911", "0.7550165711387868", "0.7695048538066305", "-0.5376291536268133", "-0.9023602768006903", "1.9588857033023839", "false"},
      {"train_030", "train", "1.7661557005385151", "3.78951305188487E-05", "1.0967411186028035", "-0.47666395962176", "0.012629202001567013", "0.8840771863600413", "0.29592826328431876", "1.965741516718844", "false"},
      {"train_031", "train", "1.2253887660464833", "4.62482296702827E-05", "1.1491543135388724", "-0.40511855692895726", "-0.3671311488892061", "-0.11876054566837145", "0.9941414109903138", "3.1493310669091104", "false"},
      {"validation_000", "validation", "1.2590884082366052", "3.12985818990909E-05", "1.07556314195433", "0.5479687966000135", "-1.1914270016535997", "-0.9422168498995661", "-0.7435410337221224", "3.03129447053778", "false"},
      {"validation_001", "validation", "1.6854024092323716", "1.13133366567427E-05", "1.2505161527448843", "1.0159097684475908", "1.31844951093136", "0.8408265426247895", "0.04245530971797424", "2.0791992388601734", "false"},
      {"validation_002", "validation", "1.736647995833823", "6.76153212789126E-05", "1.4253072957469577", "-0.9576175174245467", "0.34318011869920034", "0.12566379533280148", "0.5646530989080571", "2.0056376904235464", "false"},
      {"validation_003", "validation", "1.4717840975611896", "2.74612421219776E-05", "1.13228158323315", "0.3779676965616546", "-0.25560879843776013", "-1.2529056045244942", "0.27458040032334374", "2.4608710124182136", "false"},
      {"validation_004", "validation", "1.349566992440536", "4.13118062126926E-05", "0.9454246284500539", "-0.2837569175197371", "0.7859083886894552", "-0.5010944430274485", "1.092686119460527", "2.756830531250983", "false"},
      {"validation_005", "validation", "1.64820589391911", "1.53778987995881E-05", "1.3943954487864665", "-1.0469173924438149", "-0.6579173392939242", "-0.14599012731445135", "-0.37967153268585885", "2.1363211099825046", "false"},
      {"validation_006", "validation", "1.408625040492129", "4.94356473857333E-05", "0.8306002884534137", "-0.021350021723063807", "-0.9204423733262973", "1.0420891058660748", "0.8885660420863175", "2.6047448836829528", "false"},
      {"validation_007", "validation", "1.29365618335965", "1.90677095246137E-05", "1.0346652659059528", "0.7248603149973873", "-0.10469020335179463", "0.6570508896894014", "-0.20026729348818625", "2.919713754407342", "false"},
      {"validation_008", "validation", "1.535151090986702", "5.76329489049085E-05", "1.326734967471695", "-0.5819681283949691", "0.15768812995240392", "0.27085806764731346", "-1.1259532539312225", "2.3327090727923503", "false"},
      {"validation_009", "validation", "1.5560944052429106", "2.11980707055332E-05", "0.8784581947821548", "0.01104524129234429", "0.5080976233034907", "-0.2909148465012557", "-0.6803670976333653", "2.293443122719038", "false"},
      {"blind_000", "blind", "1.388815823919062", "1.75875483393612E-05", "0.7846548971153765", "0.9046600724478236", "-0.7616886073037004", "0.8377098384193772", "1.2073402366516746", "2.6537007172758003", "false"},
      {"blind_001", "blind", "1.4900952235702634", "7.39109404009339E-05", "1.4941235486285371", "-1.725896741419884", "0.6578570141023133", "-0.8130024903266894", "0.12918956428813225", "2.422300144842995", "false"},
      {"sealed_002", "blind", "1.45", "5.5E-05", "1.15", "0.35", "-0.55", "0.20", "-0.30", "2.508520960185669", "false"},
      {"sealed_003", "blind", "1.60", "6.5E-05", "1.35", "-0.45", "0.30", "-0.25", "0.50", "2.2155611155244963", "false"},
      {"sealed_004", "blind", "1.32", "4.5E-05", "1.25", "0.20", "0.40", "-0.60", "0.10", "2.8404458125093965", "false"},
      {"sealed_005", "blind", "1.72", "3.8E-05", "1.40", "-0.30", "-0.20", "0.45", "-0.40", "2.0289191806627773", "false"},
      {"sealed_006", "blind", "1.25", "1.2E-05", "0.80", "1.20", "-1.00", "0.90", "-0.80", "3.062179329547653", "false"},
      {"sealed_007", "blind", "1.78", "1.6E-05", "0.88", "-1.10", "1.00", "-0.90", "0.80", "1.9475997350158887", "false"},
      {"sealed_006_het", "blind_het", "1.25", "1.2E-05", "0.80", "1.20", "-1.00", "0.90", "-0.80", "3.062179329547653", "true"},
      {"sealed_007_het", "blind_het", "1.78", "1.6E-05", "0.88", "-1.10", "1.00", "-0.90", "0.80", "1.9475997350158887", "true"}
    };
    runOne("warmup", base + "warmup.csv", 1.2, 1e-5, 0.75, 0, 0, 0, 0, 3.245031, false, "coarse", false);
    Model last = null;
    for (int i = 0; i < cases.length; i++) {
      String[] c = cases[i];
      boolean het = "true".equals(c[10]);
      runOne(c[0], base + c[0] + "_complex_domain.csv",
          Double.parseDouble(c[2]), Double.parseDouble(c[3]), Double.parseDouble(c[4]),
          Double.parseDouble(c[5]), Double.parseDouble(c[6]), Double.parseDouble(c[7]),
          Double.parseDouble(c[8]), Double.parseDouble(c[9]), het, "fine", false);
    }
    for (int i = 0; i < cases.length; i++) {
      String[] c = cases[i];
      if (!"validation".equals(c[1])) continue;
      for (String tier : new String[] {"coarse", "medium"}) {
        runOne(c[0] + "_" + tier, base + c[0] + "_" + tier + "_domain.csv",
            Double.parseDouble(c[2]), Double.parseDouble(c[3]), Double.parseDouble(c[4]),
            0, 0, 0, 0, Double.parseDouble(c[9]), false, tier, false);
      }
      if ("validation_000".equals(c[0])) {
        last = runOne("validation_000_ultrafine", base + "validation_000_ultrafine_domain.csv",
            Double.parseDouble(c[2]), Double.parseDouble(c[3]), Double.parseDouble(c[4]),
            0, 0, 0, 0, Double.parseDouble(c[9]), false, "ultrafine", true);
      }
    }
    return last;
  }

  public static void main(String[] args) {
    run();
  }
}
